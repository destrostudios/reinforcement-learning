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

package org.deeplearning4j.rl4j.policy;

import lombok.AllArgsConstructor;
import org.deeplearning4j.rl4j.learning.Learning;
import org.deeplearning4j.rl4j.network.CommonOutputNames;
import org.deeplearning4j.rl4j.network.IOutputNeuralNet;
import org.deeplearning4j.rl4j.network.dqn.DQN;
import org.deeplearning4j.rl4j.network.dqn.IDQN;
import org.deeplearning4j.rl4j.space.Encodable;
import org.deeplearning4j.rl4j.observation.Observation;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.IOException;

@AllArgsConstructor
public class DQNPolicy<OBSERVATION> extends Policy<Integer> {

    final private IOutputNeuralNet neuralNet;

    public static <OBSERVATION extends Encodable> DQNPolicy<OBSERVATION> load(String path) throws IOException {
        return new DQNPolicy<>(DQN.load(path));
    }

    public IOutputNeuralNet getNeuralNet() {
        return neuralNet;
    }

    @Override
    public Integer nextAction(Observation obs) {
        INDArray output = neuralNet.output(obs).get(CommonOutputNames.QValues);
        return Learning.getMaxAction(output);
    }

    @Deprecated
    public Integer nextAction(INDArray input) {
        INDArray output = neuralNet.output(input).get(CommonOutputNames.QValues);
        return Learning.getMaxAction(output);
    }

    public void save(String filename) throws IOException {
        // TODO: refac load & save. Code below should continue to work in the meantime because it's only called by the lecacy code and it's only using a DQN network with DQNPolicy
        ((IDQN)neuralNet).save(filename);
    }

}
