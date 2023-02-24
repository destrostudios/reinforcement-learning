package com.destrostudios.rl.test.flappybird;

import org.apache.commons.lang3.builder.Builder;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.rl4j.agent.Agent;
import org.deeplearning4j.rl4j.agent.AgentLearner;
import org.deeplearning4j.rl4j.agent.IAgentLearner;
import org.deeplearning4j.rl4j.agent.learning.algorithm.dqn.BaseTransitionTDAlgorithm;
import org.deeplearning4j.rl4j.agent.learning.update.updater.NeuralNetUpdaterConfiguration;
import org.deeplearning4j.rl4j.agent.listener.AgentListener;
import org.deeplearning4j.rl4j.builder.DoubleDQNBuilder;
import org.deeplearning4j.rl4j.environment.Environment;
import org.deeplearning4j.rl4j.environment.StepResult;
import org.deeplearning4j.rl4j.experience.ReplayMemoryExperienceHandler;
import org.deeplearning4j.rl4j.network.ChannelToNetworkInputMapper;
import org.deeplearning4j.rl4j.network.ITrainableNeuralNet;
import org.deeplearning4j.rl4j.network.QNetwork;
import org.deeplearning4j.rl4j.observation.Observation;
import org.deeplearning4j.rl4j.observation.transform.TransformProcess;
import org.deeplearning4j.rl4j.observation.transform.operation.ArrayToINDArrayTransform;
import org.deeplearning4j.rl4j.policy.EpsGreedy;
import org.deeplearning4j.rl4j.trainer.ITrainer;
import org.deeplearning4j.rl4j.trainer.SyncTrainer;
import org.deeplearning4j.rl4j.util.Constants;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.rng.Random;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.util.ArrayList;
import java.util.List;

public class TestFlappyBird {

    private static final int NUM_EPISODES = 10000;

    private static final ChannelToNetworkInputMapper.NetworkInputToChannelBinding[] INPUT_CHANNEL_BINDINGS = new ChannelToNetworkInputMapper.NetworkInputToChannelBinding[] {
            ChannelToNetworkInputMapper.NetworkInputToChannelBinding.map("mydata-in", "mydata")
    };
    private static final String[] TRANSFORM_PROCESS_OUTPUT_CHANNELS = new String[] { "mydata" };

    public static void main(String[] args) {
        Random rnd = Nd4j.getRandomFactory().getNewRandomInstance(123);
        Builder<Environment<Integer>> environmentBuilder = () -> {
            FlappyBird flappyBird = new FlappyBird(rnd);
            new GameWindow(flappyBird).setVisible(true);
            return flappyBird;
        };
        Builder<TransformProcess> transformProcessBuilder = () -> TransformProcess.builder()
                .transform("mydata", new ArrayToINDArrayTransform())
                .build(TRANSFORM_PROCESS_OUTPUT_CHANNELS);

        List<AgentListener<Integer>> listeners = new ArrayList<AgentListener<Integer>>() {
            {
                add(new EpisodeScorePrinter());
            }
        };

        Builder<IAgentLearner<Integer>> builder = setupDoubleDQN(environmentBuilder, transformProcessBuilder, listeners, rnd);

        ITrainer trainer = SyncTrainer.<Integer>builder()
                .agentLearnerBuilder(builder)
                .stoppingCondition(t -> t.getEpisodeCount() >= NUM_EPISODES)
                .build();

        long before = System.nanoTime();
        trainer.train();
        long after = System.nanoTime();

        System.out.println(String.format("Total time for %d episodes: %fms", NUM_EPISODES, (after - before) / 1e6));
    }

    private static Builder<IAgentLearner<Integer>> setupDoubleDQN(Builder<Environment<Integer>> environmentBuilder,
                                                                  Builder<TransformProcess> transformProcessBuilder,
                                                                  List<AgentListener<Integer>> listeners,
                                                                  Random rnd) {
        ITrainableNeuralNet network = buildQNetwork();

        DoubleDQNBuilder.Configuration configuration = DoubleDQNBuilder.Configuration.builder()
                .policyConfiguration(EpsGreedy.Configuration.builder()
                        .epsilonNbStep(3000)
                        .minEpsilon(0)
                        .build())
                .experienceHandlerConfiguration(ReplayMemoryExperienceHandler.Configuration.builder()
                        .maxReplayMemorySize(10000)
                        .batchSize(64)
                        .build())
                .neuralNetUpdaterConfiguration(NeuralNetUpdaterConfiguration.builder()
                        .targetUpdateFrequency(50)
                        .build())
                .updateAlgorithmConfiguration(BaseTransitionTDAlgorithm.Configuration.builder()
                        .gamma(0.99)
                        .build())
                .agentLearnerConfiguration(AgentLearner.Configuration.builder()
                        .build())
                .agentLearnerListeners(listeners)
                .build();
        return new DoubleDQNBuilder(configuration, network, environmentBuilder, transformProcessBuilder, rnd);
    }

    private static ComputationGraphConfiguration.GraphBuilder buildBaseNetworkConfiguration() {
        return new NeuralNetConfiguration.Builder().seed(Constants.NEURAL_NET_SEED)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(new Adam())
                .weightInit(WeightInit.XAVIER)
                .graphBuilder()
                .setInputTypes(
                    InputType.feedForward(6)
                )
                .addInputs("mydata-in")

                .layer("dl_1", new DenseLayer.Builder().activation(Activation.RELU).nOut(40).build(), "mydata-in")
                .layer("dl_out", new DenseLayer.Builder().activation(Activation.RELU).nOut(40).build(), "dl_1");
    }

    private static ITrainableNeuralNet buildQNetwork() {
        ComputationGraphConfiguration conf = buildBaseNetworkConfiguration()
                .addLayer("output", new OutputLayer.Builder(LossFunctions.LossFunction.MSE).activation(Activation.IDENTITY)
                        .nOut(FlappyBird.ACTIONS_COUNT).build(), "dl_out")

                .setOutputs("output")
                .build();

        ComputationGraph model = new ComputationGraph(conf);
        model.init();
        return QNetwork.builder()
                .withNetwork(model)
                .inputBindings(INPUT_CHANNEL_BINDINGS)
                .channelNames(TRANSFORM_PROCESS_OUTPUT_CHANNELS)
                .build();
    }

    private static class EpisodeScorePrinter implements AgentListener<Integer> {
        private int episodeCount;

        @Override
        public ListenerResponse onBeforeEpisode(Agent agent) {
            return ListenerResponse.CONTINUE;
        }

        @Override
        public ListenerResponse onBeforeStep(Agent agent, Observation observation, Integer integer) {
            return ListenerResponse.CONTINUE;
        }

        @Override
        public ListenerResponse onAfterStep(Agent agent, StepResult stepResult) {
            return ListenerResponse.CONTINUE;
        }

        @Override
        public void onAfterEpisode(Agent agent) {
            FlappyBird environment = (FlappyBird) agent.getEnvironment();
            System.out.println(String.format("[%s] Episode %4d : reward = %4.2f, score = %d, highscore = %d", agent.getId(), episodeCount, agent.getReward(), environment.getScore(), environment.getHighscore()));
            ++episodeCount;
        }
    }
}