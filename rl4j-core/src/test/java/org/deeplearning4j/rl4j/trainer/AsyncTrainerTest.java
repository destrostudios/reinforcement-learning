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

import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Predicate;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.Mockito.*;

@ExtendWith(MockitoExtension.class)
@Tag(TagNames.FILE_IO)
@NativeTag
public class AsyncTrainerTest {

    @Mock
    Builder<IAgentLearner<Integer>> agentLearnerBuilderMock;

    @Mock
    Predicate<AsyncTrainer<Integer>> stoppingConditionMock;

    @Mock
    IAgentLearner<Integer> agentLearnerMock;

    @BeforeEach
    public void setup() {
        when(agentLearnerBuilderMock.build()).thenReturn(agentLearnerMock);
        when(agentLearnerMock.getEpisodeStepCount()).thenReturn(100);
    }

    @Test
    public void when_ctorIsCalledWithInvalidNumberOfThreads_expect_Exception() {
        try {
            AsyncTrainer sut = new AsyncTrainer(agentLearnerBuilderMock, stoppingConditionMock, 0);
            fail("IllegalArgumentException should have been thrown");
        } catch (IllegalArgumentException exception) {
            String expectedMessage = "numThreads must be greater than 0, got:  [0]";
            String actualMessage = exception.getMessage();

            assertTrue(actualMessage.contains(expectedMessage));
        }
    }

    @Test
    public void when_runningWith2Threads_expect_2AgentLearnerCreated() {
        // Arrange
        Predicate<AsyncTrainer<Integer>> stoppingCondition = t -> true;
        AsyncTrainer sut = new AsyncTrainer(agentLearnerBuilderMock, stoppingCondition, 2);

        // Act
        sut.train();

        // Assert
        verify(agentLearnerBuilderMock, times(2)).build();
    }

    @Test
    public void when_stoppingConditionTriggered_expect_agentLearnersStopsAndCountersAreCorrect() {
        // Arrange
        AtomicInteger stoppingConditionHitCount = new AtomicInteger(0);
        Predicate<AsyncTrainer<Integer>> stoppingCondition = t -> stoppingConditionHitCount.incrementAndGet() >= 5;
        AsyncTrainer<Integer> sut = new AsyncTrainer<Integer>(agentLearnerBuilderMock, stoppingCondition, 2);

        // Act
        sut.train();

        // Assert
        assertEquals(6, stoppingConditionHitCount.get());
        assertEquals(6, sut.getEpisodeCount());
        assertEquals(600, sut.getStepCount());
    }

    @Test
    public void when_training_expect_countsAreReset() {
        // Arrange
        AtomicInteger stoppingConditionHitCount = new AtomicInteger(0);
        Predicate<AsyncTrainer<Integer>> stoppingCondition = t -> stoppingConditionHitCount.incrementAndGet() >= 5;
        AsyncTrainer<Integer> sut = new AsyncTrainer<Integer>(agentLearnerBuilderMock, stoppingCondition, 2);

        // Act
        sut.train();
        stoppingConditionHitCount.set(0);
        sut.train();

        // Assert
        assertEquals(6, sut.getEpisodeCount());
        assertEquals(600, sut.getStepCount());
    }
}
