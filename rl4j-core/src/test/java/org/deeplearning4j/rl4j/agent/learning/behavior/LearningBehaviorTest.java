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

package org.deeplearning4j.rl4j.agent.learning.behavior;

import org.deeplearning4j.rl4j.agent.learning.update.IUpdateRule;
import org.deeplearning4j.rl4j.experience.ExperienceHandler;
import org.deeplearning4j.rl4j.observation.Observation;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.runner.RunWith;
import org.mockito.ArgumentCaptor;
import org.mockito.Mock;
import org.mockito.junit.MockitoJUnitRunner;
import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.mockito.Mockito.*;

@RunWith(MockitoJUnitRunner.class)
@Tag(TagNames.FILE_IO)
@NativeTag
public class LearningBehaviorTest {

    @Mock
    ExperienceHandler<Integer, Object> experienceHandlerMock;

    @Mock
    IUpdateRule<Object> updateRuleMock;

    LearningBehavior<Integer, Object> sut;

    @BeforeEach
    public void setup() {
        sut = LearningBehavior.<Integer, Object>builder()
            .experienceHandler(experienceHandlerMock)
            .updateRule(updateRuleMock)
            .build();
    }

    @Test
    public void when_callingHandleEpisodeStart_expect_experienceHandlerResetCalled() {
        // Arrange
        LearningBehavior<Integer, Object> sut = LearningBehavior.<Integer, Object>builder()
                .experienceHandler(experienceHandlerMock)
                .updateRule(updateRuleMock)
                .build();

        // Act
        sut.handleEpisodeStart();

        // Assert
        verify(experienceHandlerMock, times(1)).reset();
    }

    @Test
    public void when_callingHandleNewExperience_expect_experienceHandlerAddExperienceCalled() {
        // Arrange
        INDArray observationData = Nd4j.rand(1, 1);
        when(experienceHandlerMock.isTrainingBatchReady()).thenReturn(false);

        // Act
        sut.handleNewExperience(new Observation(observationData), 1, 2.0, false);

        // Assert
        ArgumentCaptor<Observation> observationCaptor = ArgumentCaptor.forClass(Observation.class);
        ArgumentCaptor<Integer> actionCaptor = ArgumentCaptor.forClass(Integer.class);
        ArgumentCaptor<Double> rewardCaptor = ArgumentCaptor.forClass(Double.class);
        ArgumentCaptor<Boolean> isTerminatedCaptor = ArgumentCaptor.forClass(Boolean.class);
        verify(experienceHandlerMock, times(1)).addExperience(observationCaptor.capture(), actionCaptor.capture(), rewardCaptor.capture(), isTerminatedCaptor.capture());

        assertEquals(observationData.getDouble(0, 0), observationCaptor.getValue().getData().getDouble(0, 0), 0.00001);
        assertEquals(1, (int)actionCaptor.getValue());
        assertEquals(2.0, (double)rewardCaptor.getValue(), 0.00001);
        assertFalse(isTerminatedCaptor.getValue());

        verify(updateRuleMock, never()).update(any(List.class));
    }

    @Test
    public void when_callingHandleNewExperienceAndTrainingBatchIsReady_expect_updateRuleUpdateWithTrainingBatch() {
        // Arrange
        INDArray observationData = Nd4j.rand(1, 1);
        when(experienceHandlerMock.isTrainingBatchReady()).thenReturn(true);
        List<Object> trainingBatch = new ArrayList<Object>();
        when(experienceHandlerMock.generateTrainingBatch()).thenReturn(trainingBatch);

        // Act
        sut.handleNewExperience(new Observation(observationData), 1, 2.0, false);

        // Assert
        verify(updateRuleMock, times(1)).update(trainingBatch);
    }

    @Test
    public void when_callingHandleEpisodeEnd_expect_experienceHandlerSetFinalObservationCalled() {
        // Arrange
        INDArray observationData = Nd4j.rand(1, 1);
        when(experienceHandlerMock.isTrainingBatchReady()).thenReturn(false);

        // Act
        sut.handleEpisodeEnd(new Observation(observationData));

        // Assert
        ArgumentCaptor<Observation> observationCaptor = ArgumentCaptor.forClass(Observation.class);
        verify(experienceHandlerMock, times(1)).setFinalObservation(observationCaptor.capture());

        assertEquals(observationData.getDouble(0, 0), observationCaptor.getValue().getData().getDouble(0, 0), 0.00001);

        verify(updateRuleMock, never()).update(any(List.class));
    }

    @Test
    public void when_callingHandleEpisodeEndAndTrainingBatchIsNotEmpty_expect_updateRuleUpdateWithTrainingBatch() {
        // Arrange
        INDArray observationData = Nd4j.rand(1, 1);
        when(experienceHandlerMock.isTrainingBatchReady()).thenReturn(true);
        List<Object> trainingBatch = new ArrayList<Object>();
        when(experienceHandlerMock.generateTrainingBatch()).thenReturn(trainingBatch);

        // Act
        sut.handleEpisodeEnd(new Observation(observationData));

        // Assert
        ArgumentCaptor<Observation> observationCaptor = ArgumentCaptor.forClass(Observation.class);
        verify(experienceHandlerMock, times(1)).setFinalObservation(observationCaptor.capture());

        assertEquals(observationData.getDouble(0, 0), observationCaptor.getValue().getData().getDouble(0, 0), 0.00001);

        verify(updateRuleMock, times(1)).update(trainingBatch);
    }

    @Test
    public void when_notifyBeforeStepAndBatchUnchanged_expect_notifyNewBatchStartedNotCalled() {
        // Arrange

        // Act
        sut.notifyBeforeStep();

        // Assert
        verify(updateRuleMock, never()).notifyNewBatchStarted();
    }

    @Test
    public void when_notifyBeforeStepAndBatchChanged_expect_notifyNewBatchStartedCalledOnce() {
        // Arrange
        when(experienceHandlerMock.isTrainingBatchReady()).thenReturn(true);

        // Act
        sut.handleNewExperience(null, 0, 0, false); // mark as batch has changed
        sut.notifyBeforeStep(); // Should call notify
        sut.notifyBeforeStep(); // Should not call notify

        // Assert
        verify(updateRuleMock, times(1)).notifyNewBatchStarted();
    }

}
