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

package org.deeplearning4j.rl4j.trainer;

import org.apache.commons.lang3.builder.Builder;
import org.deeplearning4j.rl4j.agent.IAgentLearner;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.junit.runner.RunWith;
import org.mockito.Mock;
import org.mockito.junit.MockitoJUnitRunner;
import org.mockito.junit.jupiter.MockitoExtension;
import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.common.tests.tags.TagNames;

import java.util.function.Predicate;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.mockito.Mockito.*;

@ExtendWith(MockitoExtension.class)
@Tag(TagNames.FILE_IO)
@NativeTag
public class SyncTrainerTest {

    @Mock
    IAgentLearner agentLearnerMock;

    @Mock
    Builder<IAgentLearner> agentLearnerBuilder;

    SyncTrainer sut;

    public void setup(Predicate<SyncTrainer> stoppingCondition) {
        when(agentLearnerBuilder.build()).thenReturn(agentLearnerMock);
        when(agentLearnerMock.getEpisodeStepCount()).thenReturn(10);

        sut = new SyncTrainer(agentLearnerBuilder, stoppingCondition);
    }

    @Test
    public void when_training_expect_stoppingConditionWillStopTraining() {
        // Arrange
        Predicate<SyncTrainer> stoppingCondition = t -> t.getEpisodeCount() >= 5; // Stop after 5 episodes
        setup(stoppingCondition);

        // Act
        sut.train();

        // Assert
        assertEquals(5, sut.getEpisodeCount());
    }

    @Test
    public void when_training_expect_agentIsRun() {
        // Arrange
        Predicate<SyncTrainer> stoppingCondition = t -> t.getEpisodeCount() >= 5; // Stop after 5 episodes
        setup(stoppingCondition);

        // Act
        sut.train();

        // Assert
        verify(agentLearnerMock, times(5)).run();
    }

    @Test
    public void when_training_expect_countsAreReset() {
        // Arrange
        Predicate<SyncTrainer> stoppingCondition = t -> t.getEpisodeCount() >= 5; // Stop after 5 episodes
        setup(stoppingCondition);

        // Act
        sut.train();
        sut.train();

        // Assert
        assertEquals(5, sut.getEpisodeCount());
        assertEquals(50, sut.getStepCount());
    }

}
