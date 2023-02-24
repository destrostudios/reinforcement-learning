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

import org.deeplearning4j.rl4j.agent.learning.update.FeaturesLabels;
import org.deeplearning4j.rl4j.experience.StateActionReward;
import org.deeplearning4j.rl4j.network.CommonLabelNames;
import org.deeplearning4j.rl4j.network.CommonOutputNames;
import org.deeplearning4j.rl4j.network.ITrainableNeuralNet;
import org.deeplearning4j.rl4j.network.NeuralNetOutput;
import org.deeplearning4j.rl4j.observation.Observation;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.junit.runner.RunWith;
import org.mockito.ArgumentCaptor;
import org.mockito.Mock;
import org.mockito.junit.MockitoJUnitRunner;
import org.mockito.junit.jupiter.MockitoExtension;
import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.mockito.Mockito.*;

@ExtendWith(MockitoExtension.class)
@Tag(TagNames.FILE_IO)
@NativeTag
public class RecurrentAdvantageActorCriticTest {
    private static final int ACTION_SPACE_SIZE = 2;
    private static final double GAMMA = 0.99;

    @Mock
    ITrainableNeuralNet threadCurrentMock;

    @Mock
    AdvantageActorCritic.Configuration configurationMock;

    @Mock
    NeuralNetOutput neuralNetOutputMock;

    private AdvantageActorCritic sut;

    @BeforeEach
    public void init() {
        when(neuralNetOutputMock.get(CommonOutputNames.ActorCritic.Value)).thenReturn(Nd4j.create(new double[] { 123.0 }));
        when(configurationMock.getGamma()).thenReturn(GAMMA);
        when(threadCurrentMock.isRecurrent()).thenReturn(true);

        sut = new AdvantageActorCritic(threadCurrentMock, ACTION_SPACE_SIZE, configurationMock);
    }

    @Test
    public void when_observationIsTerminal_expect_initialRIsZero() {
        // Arrange
        int action = 0;
        final INDArray data = Nd4j.zeros(1, 2, 1);
        final Observation observation = new Observation(data);
        List<StateActionReward<Integer>> experience = new ArrayList<StateActionReward<Integer>>() {
            {
                add(new StateActionReward<Integer>(observation, action, 0.0, true));
            }
        };
        when(threadCurrentMock.output(observation)).thenReturn(neuralNetOutputMock);

        // Act
        sut.compute(experience);

        // Assert
        ArgumentCaptor<FeaturesLabels> argument = ArgumentCaptor.forClass(FeaturesLabels.class);
        verify(threadCurrentMock, times(1)).computeGradients(argument.capture());

        FeaturesLabels featuresLabels = argument.getValue();
        assertEquals(0.0, featuresLabels.getLabels(CommonLabelNames.ActorCritic.Value).getDouble(0), 0.000001);
    }

    @Test
    public void when_observationNonTerminal_expect_initialRIsGammaTimesOutputOfValue() {
        // Arrange
        int action = 0;
        final INDArray data = Nd4j.zeros(1, 2, 1);
        final Observation observation = new Observation(data);
        List<StateActionReward<Integer>> experience = new ArrayList<StateActionReward<Integer>>() {
            {
                add(new StateActionReward<Integer>(observation, action, 0.0, false));
            }
        };
        when(threadCurrentMock.output(observation)).thenReturn(neuralNetOutputMock);

        // Act
        sut.compute(experience);

        // Assert
        ArgumentCaptor<FeaturesLabels> argument = ArgumentCaptor.forClass(FeaturesLabels.class);
        verify(threadCurrentMock, times(1)).computeGradients(argument.capture());

        FeaturesLabels featuresLabels = argument.getValue();
        assertEquals(0.0 + GAMMA * 123.0, featuresLabels.getLabels(CommonLabelNames.ActorCritic.Value).getDouble(0), 0.00001);
    }

    @Test
    public void when_callingCompute_expect_valueAndPolicyComputedCorrectly() {
        // Arrange
        int action = 0;
        when(threadCurrentMock.output(any(Observation.class))).thenAnswer(invocation -> {
            NeuralNetOutput result = new NeuralNetOutput();
            result.put(CommonOutputNames.ActorCritic.Value, invocation.getArgument(0, Observation.class).getData().get(NDArrayIndex.point(0), NDArrayIndex.point(0), NDArrayIndex.all()).mul(-1.0));
            result.put(CommonOutputNames.ActorCritic.Policy, invocation.getArgument(0, Observation.class).getData().mul(-0.1));
            return result;
        });
        List<StateActionReward<Integer>> experience = new ArrayList<StateActionReward<Integer>>() {
            {
                add(new StateActionReward<Integer>(new Observation(Nd4j.create(new double[] { -1.1, -1.2 }).reshape(1, 2, 1)), 0, 1.0, false));
                add(new StateActionReward<Integer>(new Observation(Nd4j.create(new double[] { -2.1, -2.2 }).reshape(1, 2, 1)), 1, 2.0, false));
            }
        };

        // Act
        sut.compute(experience);

        // Assert
        ArgumentCaptor<FeaturesLabels> argument = ArgumentCaptor.forClass(FeaturesLabels.class);
        verify(threadCurrentMock, times(1)).computeGradients(argument.capture());

        // input side -- should be a stack of observations
        INDArray featuresValues = argument.getValue().getFeatures().get(0);
        assertEquals(-1.1, featuresValues.getDouble(0, 0, 0), 0.00001);
        assertEquals(-1.2, featuresValues.getDouble(0, 1, 0), 0.00001);
        assertEquals(-2.1, featuresValues.getDouble(0, 0, 1), 0.00001);
        assertEquals(-2.2, featuresValues.getDouble(0, 1, 1), 0.00001);

        // Value
        INDArray valueLabels = argument.getValue().getLabels(CommonLabelNames.ActorCritic.Value);
        assertEquals(1.0 + GAMMA * (2.0 + GAMMA * 2.1), valueLabels.getDouble(0, 0, 0), 0.00001);
        assertEquals(2.0 + GAMMA * 2.1, valueLabels.getDouble(0, 0, 1), 0.00001);

        // Policy
        INDArray policyLabels = argument.getValue().getLabels(CommonLabelNames.ActorCritic.Policy);
        assertEquals((1.0 + GAMMA * (2.0 + GAMMA * 2.1)) - 1.1, policyLabels.getDouble(0, 0, 0), 0.00001);
        assertEquals((2.0 + GAMMA * 2.1) - 2.1, policyLabels.getDouble(0, 1, 1), 0.00001);

    }
}
