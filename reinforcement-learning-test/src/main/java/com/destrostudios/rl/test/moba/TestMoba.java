package com.destrostudios.rl.test.moba;

import org.apache.commons.lang3.builder.Builder;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.rl4j.agent.LearningAgent;
import org.deeplearning4j.rl4j.agent.SteppingAgent;
import org.deeplearning4j.rl4j.agent.learning.algorithm.dqn.BaseTransitionTDAlgorithm;
import org.deeplearning4j.rl4j.agent.learning.history.DefaultHistoryProcessor;
import org.deeplearning4j.rl4j.agent.learning.history.HistoryProcessor;
import org.deeplearning4j.rl4j.agent.learning.update.updater.NeuralNetUpdaterConfiguration;
import org.deeplearning4j.rl4j.agent.listener.AgentListener;
import org.deeplearning4j.rl4j.builder.DoubleDQNBuilder;
import org.deeplearning4j.rl4j.environment.Environment;
import org.deeplearning4j.rl4j.environment.StepResult;
import org.deeplearning4j.rl4j.environment.action.IntegerAction;
import org.deeplearning4j.rl4j.environment.observation.Observation;
import org.deeplearning4j.rl4j.environment.observation.transform.TransformProcess;
import org.deeplearning4j.rl4j.environment.observation.transform.operation.ArrayToINDArrayTransform;
import org.deeplearning4j.rl4j.experience.ReplayMemoryExperienceHandler;
import org.deeplearning4j.rl4j.network.ChannelToNetworkInputMapper;
import org.deeplearning4j.rl4j.network.QNetwork;
import org.deeplearning4j.rl4j.network.TrainableNeuralNet;
import org.deeplearning4j.rl4j.policy.EpsGreedy;
import org.deeplearning4j.rl4j.trainer.SyncTrainer;
import org.deeplearning4j.rl4j.trainer.Trainer;
import org.deeplearning4j.rl4j.util.Constants;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.rng.Random;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.util.ArrayList;
import java.util.List;

public class TestMoba {

    private static final int NUM_EPISODES = 10000;

    private static final ChannelToNetworkInputMapper.NetworkInputToChannelBinding[] INPUT_CHANNEL_BINDINGS = new ChannelToNetworkInputMapper.NetworkInputToChannelBinding[] {
            ChannelToNetworkInputMapper.NetworkInputToChannelBinding.map("mydata-in", "mydata")
    };
    private static final String[] TRANSFORM_PROCESS_OUTPUT_CHANNELS = new String[] { "mydata" };

    public static void main(String[] args) {
        Random rnd = Nd4j.getRandomFactory().getNewRandomInstance(123);
        Builder<Environment<IntegerAction>> environmentBuilder = () -> new MobaEnv(rnd);
        Builder<TransformProcess> transformProcessBuilder = () -> TransformProcess.builder()
                .transform("mydata", new ArrayToINDArrayTransform())
                .build(TRANSFORM_PROCESS_OUTPUT_CHANNELS);
        Builder<HistoryProcessor<Observation>> defaultHistoryProcessorBuilder = DefaultHistoryProcessor::new;

        List<AgentListener<Observation, IntegerAction>> listeners = new ArrayList<AgentListener<Observation, IntegerAction>>() {
            {
                add(new EpisodeScorePrinter(100));
            }
        };

        TrainableNeuralNet network = buildQNetwork();
        //TrainableNeuralNet network = buildActorCriticNetwork();

        Builder<LearningAgent<Observation, IntegerAction>> builder = setupDoubleDQN(network,environmentBuilder, transformProcessBuilder, defaultHistoryProcessorBuilder,listeners, rnd);

        Trainer trainer = SyncTrainer.<Observation,IntegerAction>builder()
                .agentLearnerBuilder(builder)
                .stoppingCondition(t -> t.getEpisodeCount() >= NUM_EPISODES)
                .build();

        long before = System.nanoTime();
        trainer.train();
        long after = System.nanoTime();


        System.out.println(String.format("Total time for %d episodes: %fms", NUM_EPISODES, (after - before) / 1e6));
    }

    private static Builder<LearningAgent<Observation, IntegerAction>> setupDoubleDQN(TrainableNeuralNet network, Builder<Environment<IntegerAction>> environmentBuilder,
                                                                                     Builder<TransformProcess> transformProcessBuilder, Builder<HistoryProcessor<Observation>> defaultHistoryProcessorBuilder,
                                                                                     List<AgentListener<Observation, IntegerAction>> listeners,
                                                                                     Random rnd) {

        DoubleDQNBuilder.Configuration configuration = DoubleDQNBuilder.Configuration.<Observation, IntegerAction>builder()
                .policyConfiguration(EpsGreedy.Configuration.builder()
                        .epsilonNbStep(5000)
                        .minEpsilon(0.05f)
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
                .agentLearnerConfiguration(LearningAgent.Configuration.builder()
                        .maxEpisodeSteps(2000)
                        .build())
                .agentLearnerListeners(listeners)
                .build();
        return new DoubleDQNBuilder(configuration, network, environmentBuilder, transformProcessBuilder, defaultHistoryProcessorBuilder, rnd);
    }

    private static ComputationGraphConfiguration.GraphBuilder buildBaseNetworkConfiguration() {
        return new NeuralNetConfiguration.Builder().seed(Constants.NEURAL_NET_SEED)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(new Adam())
                .weightInit(WeightInit.XAVIER)
                .graphBuilder()
                .setInputTypes(
                    InputType.feedForward(4 + (5 * 5))
                )
                .addInputs("mydata-in")

                .layer("dl_1", new DenseLayer.Builder().activation(Activation.RELU).nOut(40).build(), "mydata-in")
                .layer("dl_out", new DenseLayer.Builder().activation(Activation.RELU).nOut(40).build(), "dl_1");
    }

    private static TrainableNeuralNet buildQNetwork() {
        ComputationGraphConfiguration conf = buildBaseNetworkConfiguration()
                .addLayer("output", new OutputLayer.Builder(LossFunctions.LossFunction.MSE).activation(Activation.IDENTITY)
                        .nOut(MobaEnv.ACTIONS_COUNT).build(), "dl_out")

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

    private static class EpisodeScorePrinter implements AgentListener<Observation, IntegerAction> {
        private final int trailingNum;
        private final boolean[] results;
        private int episodeCount;

        public EpisodeScorePrinter(int trailingNum) {
            this.trailingNum = trailingNum;
            results = new boolean[trailingNum];
        }

        @Override
        public ListenerResponse onBeforeEpisode(SteppingAgent agent) {
            return ListenerResponse.CONTINUE;
        }

        @Override
        public ListenerResponse onBeforeStep(SteppingAgent agent, Observation observation, IntegerAction integerAction) {
            return ListenerResponse.CONTINUE;
        }

        @Override
        public ListenerResponse onAfterStep(SteppingAgent agent, StepResult stepResult) {
            return ListenerResponse.CONTINUE;
        }

        @Override
        public void onAfterEpisode(SteppingAgent agent) {
            MobaEnv environment = (MobaEnv) agent.getEnvironment();
            boolean isSuccess = false;

            String result;
            if (environment.getMap().getTower2().isDead()) {
                result = "WON ******";
                isSuccess = true;
            } else if(environment.getMap().getTower1().isDead()) {
                result = "LOST";
            } else {
                result = "DID NOT FINISH";
            }

            results[episodeCount % trailingNum] = isSuccess;

            if(episodeCount >= trailingNum) {
                int successCount = 0;
                for (int i = 0; i < trailingNum; ++i) {
                    successCount += results[i] ? 1 : 0;
                }
                double successRatio = successCount / (double)trailingNum;

                System.out.println(String.format("[%s] Episode %4d : score = %4.2f, success ratio = %4.2f, result = %s", agent.getId(), episodeCount, agent.getReward(), successRatio, result));
            } else {
                System.out.println(String.format("[%s] Episode %4d : score = %4.2f, result = %s", agent.getId(), episodeCount, agent.getReward(), result));
            }
            System.out.println(environment.getMap().getAsciiImage());

            ++episodeCount;
        }
    }
}