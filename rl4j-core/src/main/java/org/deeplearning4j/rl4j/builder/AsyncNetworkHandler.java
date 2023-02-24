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

import lombok.Getter;
import org.deeplearning4j.rl4j.network.ITrainableNeuralNet;

public class AsyncNetworkHandler implements INetworksHandler {

    @Getter
    final ITrainableNeuralNet targetNetwork;

    @Getter
    ITrainableNeuralNet threadCurrentNetwork;

    @Getter
    final ITrainableNeuralNet globalCurrentNetwork;

    public AsyncNetworkHandler(ITrainableNeuralNet network) {
        globalCurrentNetwork = network;
        targetNetwork = network.clone();
    }

    @Override
    public void resetForNewBuild() {
        threadCurrentNetwork = globalCurrentNetwork.clone();
    }
}
