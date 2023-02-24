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

package org.deeplearning4j.rl4j.experience;

import org.deeplearning4j.rl4j.learning.sync.IExpReplay;
import org.deeplearning4j.rl4j.observation.Observation;
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
import org.nd4j.linalg.factory.Nd4j;

import java.util.List;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.Mockito.*;

@ExtendWith(MockitoExtension.class)
@Tag(TagNames.FILE_IO)
@NativeTag
public class ReplayMemoryExperienceHandlerTest {

    @Mock
    IExpReplay<Integer> expReplayMock;

    private ReplayMemoryExperienceHandler.Configuration buildConfiguration() {
        return ReplayMemoryExperienceHandler.Configuration.builder()
                .maxReplayMemorySize(10)
                .batchSize(5)
                .build();
    }

    @Test
    public void when_addingFirstExperience_expect_notAddedToStoreBeforeNextObservationIsAdded() {
        // Arrange
        when(expReplayMock.getDesignatedBatchSize()).thenReturn(10);

        ReplayMemoryExperienceHandler sut = new ReplayMemoryExperienceHandler(expReplayMock);

        // Act
        sut.addExperience(new Observation(Nd4j.create(new double[] { 1.0 })), 1, 1.0, false);
        boolean isStoreCalledAfterFirstAdd = mockingDetails(expReplayMock).getInvocations().stream().anyMatch(x -> x.getMethod().getName() == "store");
        sut.addExperience(new Observation(Nd4j.create(new double[] { 2.0 })), 2, 2.0, false);
        boolean isStoreCalledAfterSecondAdd = mockingDetails(expReplayMock).getInvocations().stream().anyMatch(x -> x.getMethod().getName() == "store");

        // Assert
        assertFalse(isStoreCalledAfterFirstAdd);
        assertTrue(isStoreCalledAfterSecondAdd);
    }

    @Test
    public void when_addingExperience_expect_transitionsAreCorrect() {
        // Arrange
        ReplayMemoryExperienceHandler sut = new ReplayMemoryExperienceHandler(expReplayMock);

        // Act
        sut.addExperience(new Observation(Nd4j.create(new double[] { 1.0 })), 1, 1.0, false);
        sut.addExperience(new Observation(Nd4j.create(new double[] { 2.0 })), 2, 2.0, false);
        sut.setFinalObservation(new Observation(Nd4j.create(new double[] { 3.0 })));

        // Assert
        ArgumentCaptor<StateActionRewardState<Integer>> argument = ArgumentCaptor.forClass(StateActionRewardState.class);
        verify(expReplayMock, times(2)).store(argument.capture());
        List<StateActionRewardState<Integer>> stateActionRewardStates = argument.getAllValues();

        assertEquals(1.0, stateActionRewardStates.get(0).getObservation().getData().getDouble(0), 0.00001);
        assertEquals(1, (int) stateActionRewardStates.get(0).getAction());
        assertEquals(1.0, stateActionRewardStates.get(0).getReward(), 0.00001);
        assertEquals(2.0, stateActionRewardStates.get(0).getNextObservation().getChannelData(0).getDouble(0), 0.00001);

        assertEquals(2.0, stateActionRewardStates.get(1).getObservation().getData().getDouble(0), 0.00001);
        assertEquals(2, (int) stateActionRewardStates.get(1).getAction());
        assertEquals(2.0, stateActionRewardStates.get(1).getReward(), 0.00001);
        assertEquals(3.0, stateActionRewardStates.get(1).getNextObservation().getChannelData(0).getDouble(0), 0.00001);

    }

    @Test
    public void when_settingFinalObservation_expect_nextAddedExperienceDoNotUsePreviousObservation() {
        // Arrange
        ReplayMemoryExperienceHandler sut = new ReplayMemoryExperienceHandler(expReplayMock);

        // Act
        sut.addExperience(new Observation(Nd4j.create(new double[] { 1.0 })), 1, 1.0, false);
        sut.setFinalObservation(new Observation(Nd4j.create(new double[] { 2.0 })));
        sut.addExperience(new Observation(Nd4j.create(new double[] { 3.0 })), 3, 3.0, false);

        // Assert
        ArgumentCaptor<StateActionRewardState<Integer>> argument = ArgumentCaptor.forClass(StateActionRewardState.class);
        verify(expReplayMock, times(1)).store(argument.capture());
        StateActionRewardState<Integer> stateActionRewardState = argument.getValue();

        assertEquals(1, (int) stateActionRewardState.getAction());
    }

    @Test
    public void when_addingExperience_expect_getTrainingBatchSizeReturnSize() {
        // Arrange
        ReplayMemoryExperienceHandler sut = new ReplayMemoryExperienceHandler(buildConfiguration(), Nd4j.getRandom());
        sut.addExperience(new Observation(Nd4j.create(new double[] { 1.0 })), 1, 1.0, false);
        sut.addExperience(new Observation(Nd4j.create(new double[] { 2.0 })), 2, 2.0, false);
        sut.setFinalObservation(new Observation(Nd4j.create(new double[] { 3.0 })));

        // Act
        int size = sut.getTrainingBatchSize();

        // Assert
        assertEquals(2, size);
    }

    @Test
    public void when_experienceSizeIsSmallerThanBatchSize_expect_TrainingBatchIsNotReady() {
        // Arrange
        ReplayMemoryExperienceHandler sut = new ReplayMemoryExperienceHandler(buildConfiguration(), Nd4j.getRandom());
        sut.addExperience(new Observation(Nd4j.create(new double[] { 1.0 })), 1, 1.0, false);
        sut.addExperience(new Observation(Nd4j.create(new double[] { 2.0 })), 2, 2.0, false);
        sut.setFinalObservation(new Observation(Nd4j.create(new double[] { 3.0 })));

        // Act

        // Assert
        assertFalse(sut.isTrainingBatchReady());
    }

    @Test
    public void when_experienceSizeIsGreaterOrEqualToBatchSize_expect_TrainingBatchIsReady() {
        // Arrange
        ReplayMemoryExperienceHandler sut = new ReplayMemoryExperienceHandler(buildConfiguration(), Nd4j.getRandom());
        sut.addExperience(new Observation(Nd4j.create(new double[] { 1.0 })), 1, 1.0, false);
        sut.addExperience(new Observation(Nd4j.create(new double[] { 2.0 })), 2, 2.0, false);
        sut.addExperience(new Observation(Nd4j.create(new double[] { 3.0 })), 3, 3.0, false);
        sut.addExperience(new Observation(Nd4j.create(new double[] { 4.0 })), 4, 4.0, false);
        sut.addExperience(new Observation(Nd4j.create(new double[] { 5.0 })), 5, 5.0, false);
        sut.setFinalObservation(new Observation(Nd4j.create(new double[] { 6.0 })));

        // Act

        // Assert
        assertTrue(sut.isTrainingBatchReady());
    }

}
