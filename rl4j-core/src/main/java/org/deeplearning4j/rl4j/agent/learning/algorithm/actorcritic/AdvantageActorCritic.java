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
package org.deeplearning4j.rl4j.agent.learning.algorithm.actorcritic;

import lombok.Builder;
import lombok.Data;
import lombok.NonNull;
import lombok.experimental.SuperBuilder;
import org.deeplearning4j.rl4j.agent.learning.algorithm.IUpdateAlgorithm;
import org.deeplearning4j.rl4j.agent.learning.update.Features;
import org.deeplearning4j.rl4j.agent.learning.update.FeaturesBuilder;
import org.deeplearning4j.rl4j.agent.learning.update.FeaturesLabels;
import org.deeplearning4j.rl4j.agent.learning.update.Gradients;
import org.deeplearning4j.rl4j.experience.StateActionReward;
import org.deeplearning4j.rl4j.network.CommonLabelNames;
import org.deeplearning4j.rl4j.network.CommonOutputNames;
import org.deeplearning4j.rl4j.network.ITrainableNeuralNet;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.List;

public class AdvantageActorCritic implements IUpdateAlgorithm<Gradients, StateActionReward<Integer>> {

    private final ITrainableNeuralNet threadCurrent;

    private final double gamma;

    private final ActorCriticHelper algorithmHelper;

    private final FeaturesBuilder featuresBuilder;

    public AdvantageActorCritic(@NonNull ITrainableNeuralNet threadCurrent,
                                int actionSpaceSize,
                                @NonNull Configuration configuration) {
        this.threadCurrent = threadCurrent;
        gamma = configuration.getGamma();

        algorithmHelper = threadCurrent.isRecurrent()
                ? new RecurrentActorCriticHelper(actionSpaceSize)
                : new NonRecurrentActorCriticHelper(actionSpaceSize);

        featuresBuilder = new FeaturesBuilder(threadCurrent.isRecurrent());
    }

    @Override
    public Gradients compute(List<StateActionReward<Integer>> trainingBatch) {
        int size = trainingBatch.size();

        Features features = featuresBuilder.build(trainingBatch);
        INDArray values = algorithmHelper.createValueLabels(size);
        INDArray policy = algorithmHelper.createPolicyLabels(size);

        StateActionReward<Integer> stateActionReward = trainingBatch.get(size - 1);
        double value;
        if (stateActionReward.isTerminal()) {
            value = 0;
        } else {
            value = threadCurrent.output(trainingBatch.get(size - 1).getObservation()).get(CommonOutputNames.ActorCritic.Value).getDouble(0);
        }

        for (int i = size - 1; i >= 0; --i) {
            stateActionReward = trainingBatch.get(i);

            value = stateActionReward.getReward() + gamma * value;

            //the critic
            values.putScalar(i, value);

            //the actor
            double expectedV = threadCurrent.output(trainingBatch.get(i).getObservation()).get(CommonOutputNames.ActorCritic.Value).getDouble(0);
            double advantage = value - expectedV;
            algorithmHelper.setPolicy(policy, i, stateActionReward.getAction(), advantage);
        }

        FeaturesLabels featuresLabels = new FeaturesLabels(features);
        featuresLabels.putLabels(CommonLabelNames.ActorCritic.Value, values);
        featuresLabels.putLabels(CommonLabelNames.ActorCritic.Policy, policy);

        return threadCurrent.computeGradients(featuresLabels);
    }

    @SuperBuilder
    @Data
    public static class Configuration {
        /**
         * The discount factor (default is 0.99)
         */
        @Builder.Default
        double gamma = 0.99;
    }
}
