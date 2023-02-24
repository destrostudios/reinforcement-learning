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

package org.deeplearning4j.rl4j.builder;

import org.apache.commons.lang3.builder.Builder;
import org.deeplearning4j.rl4j.agent.AgentLearner;
import org.deeplearning4j.rl4j.agent.learning.algorithm.IUpdateAlgorithm;
import org.deeplearning4j.rl4j.agent.learning.update.updater.INeuralNetUpdater;
import org.deeplearning4j.rl4j.environment.Environment;
import org.deeplearning4j.rl4j.experience.ExperienceHandler;
import org.deeplearning4j.rl4j.network.ITrainableNeuralNet;
import org.deeplearning4j.rl4j.observation.transform.TransformProcess;
import org.deeplearning4j.rl4j.policy.IPolicy;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.junit.runner.RunWith;
import org.mockito.Mock;
import org.mockito.Mockito;
import org.mockito.junit.MockitoJUnitRunner;
import org.mockito.junit.jupiter.MockitoExtension;

import static org.mockito.Mockito.*;

@ExtendWith(MockitoExtension.class)
public class BaseAgentLearnerBuilderTest {
    @Mock
    BaseAgentLearnerBuilder.Configuration configuration;

    @Mock
    ITrainableNeuralNet neuralNet;

    @Mock
    Builder<Environment<Integer>> environmentBuilder;

    @Mock
    Builder<TransformProcess> transformProcessBuilder;

    @Mock
    IUpdateAlgorithm updateAlgorithmMock;

    @Mock
    INeuralNetUpdater neuralNetUpdaterMock;

    @Mock
    ExperienceHandler experienceHandlerMock;

    @Mock
    Environment environmentMock;

    @Mock
    IPolicy policyMock;

    @Mock
    TransformProcess transformProcessMock;

    BaseAgentLearnerBuilder sut;

    @BeforeEach
    public void setup() {
        sut = mock(
                BaseAgentLearnerBuilder.class,
                Mockito.withSettings()
                        .useConstructor(configuration, neuralNet, environmentBuilder, transformProcessBuilder)
                        .defaultAnswer(Mockito.CALLS_REAL_METHODS)
        );

        AgentLearner.Configuration agentLearnerConfiguration = AgentLearner.Configuration.builder().maxEpisodeSteps(200).build();

        when(sut.buildUpdateAlgorithm()).thenReturn(updateAlgorithmMock);
        when(sut.buildNeuralNetUpdater()).thenReturn(neuralNetUpdaterMock);
        when(sut.buildExperienceHandler()).thenReturn(experienceHandlerMock);
        when(environmentBuilder.build()).thenReturn(environmentMock);
        when(transformProcessBuilder.build()).thenReturn(transformProcessMock);
        when(sut.buildPolicy()).thenReturn(policyMock);
        when(configuration.getAgentLearnerConfiguration()).thenReturn(agentLearnerConfiguration);
    }

    @Test
    public void when_buildingAgentLearner_expect_dependenciesAndAgentLearnerIsBuilt() {
        // Arrange

        // Act
        sut.build();

        // Assert
        verify(environmentBuilder, times(1)).build();
        verify(transformProcessBuilder, times(1)).build();
        verify(sut, times(1)).buildPolicy();
        verify(sut, times(1)).buildExperienceHandler();
        verify(sut, times(1)).buildUpdateAlgorithm();
        verify(sut, times(1)).buildNeuralNetUpdater();
        verify(sut, times(1)).buildAgentLearner();
    }

}
