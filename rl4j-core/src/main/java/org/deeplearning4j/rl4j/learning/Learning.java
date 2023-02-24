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

package org.deeplearning4j.rl4j.learning;

import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.Setter;
import lombok.Value;
import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.rl4j.network.NeuralNet;
import org.deeplearning4j.rl4j.space.ActionSpace;
import org.deeplearning4j.rl4j.space.Encodable;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

@Slf4j
public abstract class Learning<OBSERVATION extends Encodable, A, AS extends ActionSpace<A>, NN extends NeuralNet>
                implements ILearning<OBSERVATION, A, AS>, NeuralNetFetchable<NN> {

    @Getter @Setter
    protected int stepCount = 0;
    @Getter @Setter
    private int epochCount = 0;
    @Getter @Setter
    private IHistoryProcessor historyProcessor = null;

    public static Integer getMaxAction(INDArray vector) {
        return Nd4j.argMax(vector, Integer.MAX_VALUE).getInt(0);
    }

    public static int[] makeShape(int size, int[] shape) {
        int[] nshape = new int[shape.length + 1];
        nshape[0] = size;
        System.arraycopy(shape, 0, nshape, 1, shape.length);
        return nshape;
    }

    public static int[] makeShape(int batch, int[] shape, int length) {
        int[] nshape = new int[3];
        nshape[0] = batch;
        nshape[1] = 1;
        for (int i = 0; i < shape.length; i++) {
            nshape[1] *= shape[i];
        }
        nshape[2] = length;
        return nshape;
    }

    public abstract NN getNeuralNet();

    public void incrementStep() {
        stepCount++;
    }

    public void incrementEpoch() {
        epochCount++;
    }

    public void setHistoryProcessor(HistoryProcessor.Configuration conf) {
        setHistoryProcessor(new HistoryProcessor(conf));
    }

    public void setHistoryProcessor(IHistoryProcessor historyProcessor) {
        this.historyProcessor = historyProcessor;
    }

    @AllArgsConstructor
    @Value
    public static class InitMdp<O> {
        int steps;
        O lastObs;
        double reward;
    }

}
