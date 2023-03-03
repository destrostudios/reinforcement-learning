/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package com.destrostudios.rl.test.rl4j.examples;

import com.destrostudios.rl.test.rl4j.mdp.DoAsISayOrDont;
import org.apache.commons.lang3.builder.Builder;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.graph.PreprocessorVertex;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.LSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.conf.preprocessor.FeedForwardToRnnPreProcessor;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.rl4j.agent.LearningAgent;
import org.deeplearning4j.rl4j.agent.learning.algorithm.actorcritic.AdvantageActorCritic;
import org.deeplearning4j.rl4j.agent.learning.history.DefaultHistoryProcessor;
import org.deeplearning4j.rl4j.agent.learning.history.HistoryProcessor;
import org.deeplearning4j.rl4j.agent.learning.update.updater.NeuralNetUpdaterConfiguration;
import org.deeplearning4j.rl4j.agent.listener.AgentListener;
import org.deeplearning4j.rl4j.agent.listener.utils.EpisodeScorePrinter;
import org.deeplearning4j.rl4j.builder.AdvantageActorCriticBuilder;
import org.deeplearning4j.rl4j.environment.Environment;
import org.deeplearning4j.rl4j.environment.action.IntegerAction;
import org.deeplearning4j.rl4j.environment.observation.Observation;
import org.deeplearning4j.rl4j.environment.observation.transform.TransformProcess;
import org.deeplearning4j.rl4j.environment.observation.transform.operation.ArrayToINDArrayTransform;
import org.deeplearning4j.rl4j.experience.ObservationActionExperienceHandler;
import org.deeplearning4j.rl4j.network.ActorCriticNetwork;
import org.deeplearning4j.rl4j.network.TrainableNeuralNet;
import org.deeplearning4j.rl4j.network.factory.ActorCriticLoss;
import org.deeplearning4j.rl4j.trainer.Trainer;
import org.deeplearning4j.rl4j.trainer.SyncTrainer;
import org.deeplearning4j.rl4j.util.Constants;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.rng.Random;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.util.ArrayList;
import java.util.List;

public class NStepRnnExample {

    private static final boolean USE_SEPARATE_NETWORKS = true;
    private static final int NUM_EPISODES = 3000;

    private static final int COMBINED_LSTM_LAYER_SIZE = 20;
    private static final int COMBINED_DL1_LAYER_SIZE = 20;
    private static final int COMBINED_DL2_LAYER_SIZE = 60;

    private static final int SEPARATE_LSTM_LAYER_SIZE = 10;
    private static final int SEPARATE_DL1_LAYER_SIZE = 10;
    private static final int SEPARATE_DL2_LAYER_SIZE = 10;

    private static final int NUM_INPUTS = 4;
    private static final int NUM_ACTIONS = 2;



    public static void main(String[] args) {

        Random rnd = Nd4j.getRandomFactory().getNewRandomInstance(123);

        Builder<Environment<IntegerAction>> environmentBuilder = () -> new DoAsISayOrDont(rnd);
        Builder<TransformProcess> transformProcessBuilder = () -> TransformProcess.builder()
                .transform("data", new ArrayToINDArrayTransform(1, NUM_INPUTS, 1))
                .build("data");
        Builder<HistoryProcessor<Observation>> defaultHistoryProcessorBuilder = DefaultHistoryProcessor::new;

        List<AgentListener<Observation, IntegerAction>> listeners = new ArrayList<AgentListener<Observation, IntegerAction>>() {
            {
                add(new EpisodeScorePrinter());
            }
        };

        TrainableNeuralNet network = USE_SEPARATE_NETWORKS
                ? buildSeparateActorCriticNetwork()
                : buildActorCriticNetwork();
        
        Builder<LearningAgent<Observation, IntegerAction>> builder = setupAdvantageActorCritic(network,environmentBuilder, transformProcessBuilder, defaultHistoryProcessorBuilder,listeners, rnd);

        Trainer trainer = SyncTrainer.<Observation,IntegerAction>builder()
                .agentLearnerBuilder(builder)
                .stoppingCondition(t -> t.getEpisodeCount() >= NUM_EPISODES)
                .build();

        long before = System.nanoTime();
        trainer.train();
        long after = System.nanoTime();

        System.out.println(String.format("Total time for %d episodes: %fms", NUM_EPISODES, (after - before) / 1e6));
    }

    private static Builder<LearningAgent<Observation, IntegerAction>> setupAdvantageActorCritic(TrainableNeuralNet network,Builder<Environment<IntegerAction>> environmentBuilder,
                                                                             Builder<TransformProcess> transformProcessBuilder,Builder<HistoryProcessor<Observation>> defaultHistoryProcessorBuilder,
                                                                             List<AgentListener<Observation, IntegerAction>> listeners,
                                                                             Random rnd) {

        AdvantageActorCriticBuilder.Configuration configuration = AdvantageActorCriticBuilder.Configuration.<Observation, IntegerAction>builder()
                .neuralNetUpdaterConfiguration(NeuralNetUpdaterConfiguration.builder()
                        .build())
                .advantageActorCriticConfiguration(AdvantageActorCritic.Configuration.builder()
                        .gamma(0.99)
                        .build())
                .experienceHandlerConfiguration(ObservationActionExperienceHandler.Configuration.builder()
                        .batchSize(20)
                        .build())
                .agentLearnerConfiguration(LearningAgent.Configuration.builder()
                        .maxEpisodeSteps(200)
                        .build())
                .agentLearnerListeners(listeners)
                .build();
        return new AdvantageActorCriticBuilder(configuration, network, environmentBuilder, transformProcessBuilder,defaultHistoryProcessorBuilder, rnd);
    }

    private static ComputationGraphConfiguration.GraphBuilder buildBaseNetworkConfiguration(int lstmLayerSize, int dl1Size, int dl2Size) {
        return new NeuralNetConfiguration.Builder().seed(Constants.NEURAL_NET_SEED)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(new Adam())
                .weightInit(WeightInit.XAVIER)
                .graphBuilder()
                .addInputs("input")
                .setInputTypes(InputType.recurrent(NUM_INPUTS))
                .addLayer("lstm", new LSTM.Builder().nOut(lstmLayerSize).activation(Activation.TANH).build(), "input")
                .addLayer("dl", new DenseLayer.Builder().nOut(dl1Size).activation(Activation.RELU).build(), "input", "lstm")
                .addLayer("dl-1", new DenseLayer.Builder().nOut(dl2Size).activation(Activation.RELU).build(), "dl")
                .addVertex("dl-rnn", new PreprocessorVertex(new FeedForwardToRnnPreProcessor()), "dl-1");
    }

    private static TrainableNeuralNet buildActorCriticNetwork() {
        ComputationGraphConfiguration valueConfiguration = buildBaseNetworkConfiguration(COMBINED_LSTM_LAYER_SIZE, COMBINED_DL1_LAYER_SIZE, COMBINED_DL2_LAYER_SIZE)
                .addLayer("value", new RnnOutputLayer.Builder(LossFunctions.LossFunction.MSE).activation(Activation.IDENTITY).nOut(1).build(), "dl-rnn", "lstm")
                .addLayer("softmax", new RnnOutputLayer.Builder(new ActorCriticLoss()).activation(Activation.SOFTMAX).nOut(NUM_ACTIONS).build(), "dl-rnn", "lstm")
                .setOutputs("value", "softmax")
                .build();

        ComputationGraph model = new ComputationGraph(valueConfiguration);
        model.init();

        return ActorCriticNetwork.builder()
                .withCombinedNetwork(model)
                .build();
    }

    private static TrainableNeuralNet buildSeparateActorCriticNetwork() {
        ComputationGraphConfiguration valueConfiguration = buildBaseNetworkConfiguration(SEPARATE_LSTM_LAYER_SIZE, SEPARATE_DL1_LAYER_SIZE, SEPARATE_DL2_LAYER_SIZE)
                .addLayer("value", new RnnOutputLayer.Builder(LossFunctions.LossFunction.MSE).activation(Activation.IDENTITY).nOut(1).build(), "dl-rnn", "lstm")
                .setOutputs("value")
                .build();

        ComputationGraphConfiguration policyConfiguration = buildBaseNetworkConfiguration(SEPARATE_LSTM_LAYER_SIZE, SEPARATE_DL1_LAYER_SIZE, SEPARATE_DL2_LAYER_SIZE)
                .addLayer("softmax", new RnnOutputLayer.Builder(new ActorCriticLoss()).activation(Activation.SOFTMAX).nOut(NUM_ACTIONS).build(), "dl-rnn", "lstm")
                .setOutputs("softmax")
                .build();

        ComputationGraph valueModel = new ComputationGraph(valueConfiguration);
        valueModel.init();

        ComputationGraph policyModel = new ComputationGraph(policyConfiguration);
        policyModel.init();

        return ActorCriticNetwork.builder()
                .withSeparateNetworks(valueModel, policyModel)
                .build();
    }
}