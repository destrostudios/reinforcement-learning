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

package org.deeplearning4j.rl4j.mdp.toy;

import lombok.Getter;
import lombok.Setter;
import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.gym.StepReply;
import org.deeplearning4j.rl4j.learning.NeuralNetFetchable;
import org.deeplearning4j.rl4j.mdp.MDP;
import org.deeplearning4j.rl4j.network.CommonOutputNames;
import org.deeplearning4j.rl4j.network.dqn.IDQN;
import org.deeplearning4j.rl4j.space.ArrayObservationSpace;
import org.deeplearning4j.rl4j.space.DiscreteSpace;
import org.deeplearning4j.rl4j.space.ObservationSpace;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;


@Slf4j
public class SimpleToy implements MDP<SimpleToyState, Integer, DiscreteSpace> {

    final private int maxStep;
    @Getter
    private DiscreteSpace actionSpace = new DiscreteSpace(2);
    @Getter
    private ObservationSpace<SimpleToyState> observationSpace = new ArrayObservationSpace(new int[] {1});
    private SimpleToyState simpleToyState;
    @Setter
    private NeuralNetFetchable<IDQN> fetchable;

    public SimpleToy(int maxStep) {
        this.maxStep = maxStep;
    }

    public void printTest(int maxStep) {
        INDArray input = Nd4j.create(maxStep, 1);
        for (int i = 0; i < maxStep; i++) {
            input.putRow(i, Nd4j.create(new SimpleToyState(i, i).toArray()));
        }
        INDArray output = fetchable.getNeuralNet().output(input).get(CommonOutputNames.QValues);
        log.info(output.toString());
    }

    public void close() {}

    @Override
    public boolean isDone() {
        return simpleToyState.getStep() == maxStep;
    }

    public SimpleToyState reset() {
        if (fetchable != null)
            printTest(maxStep);

        return simpleToyState = new SimpleToyState(0, 0);
    }

    public StepReply<SimpleToyState> step(Integer a) {
        double reward = (simpleToyState.getStep() % 2 == 0) ? 1 - a : a;
        simpleToyState = new SimpleToyState(simpleToyState.getI() + 1, simpleToyState.getStep() + 1);
        return new StepReply<>(simpleToyState, reward, isDone(), null);
    }

    public SimpleToy newInstance() {
        SimpleToy simpleToy = new SimpleToy(maxStep);
        simpleToy.setFetchable(fetchable);
        return simpleToy;
    }

}
