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

package org.deeplearning4j.rl4j.agent.learning.update.updater.async;

import org.deeplearning4j.rl4j.agent.learning.update.Gradients;
import org.deeplearning4j.rl4j.agent.learning.update.updater.NeuralNetUpdaterConfiguration;
import org.deeplearning4j.rl4j.network.ITrainableNeuralNet;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.runner.RunWith;
import org.mockito.Mock;
import org.mockito.junit.MockitoJUnitRunner;
import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.common.tests.tags.TagNames;

import static org.mockito.Mockito.*;

@RunWith(MockitoJUnitRunner.class)
@Tag(TagNames.FILE_IO)
@NativeTag
public class AsyncGradientsNeuralNetUpdaterTest {

    @Mock
    ITrainableNeuralNet threadCurrentMock;

    @Mock
    ITrainableNeuralNet globalCurrentMock;

    @Mock
    AsyncSharedNetworksUpdateHandler asyncSharedNetworksUpdateHandlerMock;

    @Test
    public void when_callingUpdate_expect_handlerCalledAndThreadCurrentUpdated() {
        // Arrange
        NeuralNetUpdaterConfiguration configuration = NeuralNetUpdaterConfiguration.builder()
                .targetUpdateFrequency(2)
                .build();
        AsyncGradientsNeuralNetUpdater sut = new AsyncGradientsNeuralNetUpdater(threadCurrentMock, asyncSharedNetworksUpdateHandlerMock);
        Gradients gradients = new Gradients(10);

        // Act
        sut.update(gradients);

        // Assert
        verify(asyncSharedNetworksUpdateHandlerMock, times(1)).handleGradients(gradients);
        verify(threadCurrentMock, never()).copyFrom(globalCurrentMock);
    }

    @Test
    public void when_synchronizeCurrentIsCalled_expect_synchronizeThreadCurrentWithGlobal() {
        // Arrange
        NeuralNetUpdaterConfiguration configuration = NeuralNetUpdaterConfiguration.builder()
                .build();
        AsyncGradientsNeuralNetUpdater sut = new AsyncGradientsNeuralNetUpdater(threadCurrentMock, asyncSharedNetworksUpdateHandlerMock);
        when(asyncSharedNetworksUpdateHandlerMock.getGlobalCurrent()).thenReturn(globalCurrentMock);

        // Act
        sut.synchronizeCurrent();

        // Assert
        verify(threadCurrentMock, times(1)).copyFrom(globalCurrentMock);
    }
}
