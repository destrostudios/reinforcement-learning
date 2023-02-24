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

package org.deeplearning4j.rl4j.learning.async.a3c.discrete;

import lombok.Getter;
import org.deeplearning4j.rl4j.learning.async.*;
import org.deeplearning4j.rl4j.learning.configuration.A3CLearningConfiguration;
import org.deeplearning4j.rl4j.learning.listener.TrainingListenerList;
import org.deeplearning4j.rl4j.mdp.MDP;
import org.deeplearning4j.rl4j.network.ac.IActorCritic;
import org.deeplearning4j.rl4j.space.Encodable;
import org.deeplearning4j.rl4j.policy.ACPolicy;
import org.deeplearning4j.rl4j.policy.Policy;
import org.deeplearning4j.rl4j.space.DiscreteSpace;
import org.nd4j.linalg.api.rng.Random;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.api.rng.Random;

public class A3CThreadDiscrete<OBSERVATION extends Encodable> extends AsyncThreadDiscrete<OBSERVATION, IActorCritic> {

    @Getter
    final protected A3CLearningConfiguration configuration;
    @Getter
    final protected IAsyncGlobal<IActorCritic> asyncGlobal;
    @Getter
    final protected int threadNumber;

    final private Random rnd;

    public A3CThreadDiscrete(MDP<OBSERVATION, Integer, DiscreteSpace> mdp, IAsyncGlobal<IActorCritic> asyncGlobal,
                             A3CLearningConfiguration a3cc, int deviceNum, TrainingListenerList listeners,
                             int threadNumber) {
        super(asyncGlobal, mdp, listeners, threadNumber, deviceNum);
        this.configuration = a3cc;
        this.asyncGlobal = asyncGlobal;
        this.threadNumber = threadNumber;

        Long seed = configuration.getSeed();
        rnd = Nd4j.getRandom();
        if (seed != null) {
            rnd.setSeed(seed + threadNumber);
        }

        setUpdateAlgorithm(buildUpdateAlgorithm());
    }

    @Override
    protected Policy<Integer> getPolicy(IActorCritic net) {
        return new ACPolicy<OBSERVATION>(net, true, rnd);
    }

    /**
     * calc the gradients based on the n-step rewards
     */
    @Override
    protected UpdateAlgorithm<IActorCritic> buildUpdateAlgorithm() {
        int[] shape = getHistoryProcessor() == null ? getMdp().getObservationSpace().getShape() : getHistoryProcessor().getConf().getShape();
        return new AdvantageActorCriticUpdateAlgorithm(asyncGlobal.getTarget().isRecurrent(), shape, getMdp().getActionSpace().getSize(), configuration.getGamma());
    }
}
