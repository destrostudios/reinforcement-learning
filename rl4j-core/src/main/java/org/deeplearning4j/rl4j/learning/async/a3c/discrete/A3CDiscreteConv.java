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

import org.deeplearning4j.rl4j.learning.HistoryProcessor;
import org.deeplearning4j.rl4j.learning.async.AsyncThread;
import org.deeplearning4j.rl4j.learning.configuration.A3CLearningConfiguration;
import org.deeplearning4j.rl4j.mdp.MDP;
import org.deeplearning4j.rl4j.network.ac.ActorCriticFactoryCompGraph;
import org.deeplearning4j.rl4j.network.ac.ActorCriticFactoryCompGraphStdConv;
import org.deeplearning4j.rl4j.network.ac.IActorCritic;
import org.deeplearning4j.rl4j.network.configuration.ActorCriticNetworkConfiguration;
import org.deeplearning4j.rl4j.space.Encodable;
import org.deeplearning4j.rl4j.space.DiscreteSpace;
import org.deeplearning4j.rl4j.util.DataManagerTrainingListener;
import org.deeplearning4j.rl4j.util.IDataManager;

public class A3CDiscreteConv<OBSERVATION extends Encodable> extends A3CDiscrete<OBSERVATION> {

    final private HistoryProcessor.Configuration hpconf;

    @Deprecated
    public A3CDiscreteConv(MDP<OBSERVATION, Integer, DiscreteSpace> mdp, IActorCritic actorCritic,
                           HistoryProcessor.Configuration hpconf, A3CConfiguration conf, IDataManager dataManager) {
        this(mdp, actorCritic, hpconf, conf);
        addListener(new DataManagerTrainingListener(dataManager));
    }

    @Deprecated
    public A3CDiscreteConv(MDP<OBSERVATION, Integer, DiscreteSpace> mdp, IActorCritic IActorCritic,
                           HistoryProcessor.Configuration hpconf, A3CConfiguration conf) {

        super(mdp, IActorCritic, conf.toLearningConfiguration());
        this.hpconf = hpconf;
        setHistoryProcessor(hpconf);
    }

    public A3CDiscreteConv(MDP<OBSERVATION, Integer, DiscreteSpace> mdp, IActorCritic IActorCritic,
                           HistoryProcessor.Configuration hpconf, A3CLearningConfiguration conf) {
        super(mdp, IActorCritic, conf);
        this.hpconf = hpconf;
        setHistoryProcessor(hpconf);
    }

    @Deprecated
    public A3CDiscreteConv(MDP<OBSERVATION, Integer, DiscreteSpace> mdp, ActorCriticFactoryCompGraph factory,
                           HistoryProcessor.Configuration hpconf, A3CConfiguration conf, IDataManager dataManager) {
        this(mdp, factory.buildActorCritic(hpconf.getShape(), mdp.getActionSpace().getSize()), hpconf, conf, dataManager);
    }

    @Deprecated
    public A3CDiscreteConv(MDP<OBSERVATION, Integer, DiscreteSpace> mdp, ActorCriticFactoryCompGraph factory,
                           HistoryProcessor.Configuration hpconf, A3CConfiguration conf) {
        this(mdp, factory.buildActorCritic(hpconf.getShape(), mdp.getActionSpace().getSize()), hpconf, conf);
    }

    public A3CDiscreteConv(MDP<OBSERVATION, Integer, DiscreteSpace> mdp, ActorCriticFactoryCompGraph factory,
                           HistoryProcessor.Configuration hpconf, A3CLearningConfiguration conf) {
        this(mdp, factory.buildActorCritic(hpconf.getShape(), mdp.getActionSpace().getSize()), hpconf, conf);
    }

    @Deprecated
    public A3CDiscreteConv(MDP<OBSERVATION, Integer, DiscreteSpace> mdp, ActorCriticFactoryCompGraphStdConv.Configuration netConf,
                           HistoryProcessor.Configuration hpconf, A3CConfiguration conf, IDataManager dataManager) {
        this(mdp, new ActorCriticFactoryCompGraphStdConv(netConf.toNetworkConfiguration()), hpconf, conf, dataManager);
    }

    @Deprecated
    public A3CDiscreteConv(MDP<OBSERVATION, Integer, DiscreteSpace> mdp, ActorCriticFactoryCompGraphStdConv.Configuration netConf,
                           HistoryProcessor.Configuration hpconf, A3CConfiguration conf) {
        this(mdp, new ActorCriticFactoryCompGraphStdConv(netConf.toNetworkConfiguration()), hpconf, conf);
    }

    public A3CDiscreteConv(MDP<OBSERVATION, Integer, DiscreteSpace> mdp, ActorCriticNetworkConfiguration netConf,
                           HistoryProcessor.Configuration hpconf, A3CLearningConfiguration conf) {
        this(mdp, new ActorCriticFactoryCompGraphStdConv(netConf), hpconf, conf);
    }

    @Override
    public AsyncThread newThread(int i, int deviceNum) {
        AsyncThread at = super.newThread(i, deviceNum);
        at.setHistoryProcessor(hpconf);
        return at;
    }
}
