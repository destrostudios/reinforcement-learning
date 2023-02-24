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
package org.deeplearning4j.rl4j.agent.learning.algorithm.actorcritic;

import org.deeplearning4j.rl4j.helper.INDArrayHelper;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class NonRecurrentActorCriticHelper extends ActorCriticHelper {
    private final int actionSpaceSize;

    public NonRecurrentActorCriticHelper(int actionSpaceSize) {
        this.actionSpaceSize = actionSpaceSize;
    }

    @Override
    public INDArray createValueLabels(int trainingBatchSize) {
        return Nd4j.create(trainingBatchSize, 1);
    }

    @Override
    public INDArray createPolicyLabels(int trainingBatchSize) {
        return Nd4j.zeros(trainingBatchSize, actionSpaceSize);
    }

    @Override
    public void setPolicy(INDArray policy, long idx, int action, double advantage) {
        policy.putScalar(idx, action, advantage);
    }
}