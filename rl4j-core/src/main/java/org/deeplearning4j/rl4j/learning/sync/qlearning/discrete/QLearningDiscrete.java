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

package org.deeplearning4j.rl4j.learning.sync.qlearning.discrete;

import lombok.Getter;
import org.deeplearning4j.gym.StepReply;
import org.deeplearning4j.rl4j.agent.learning.algorithm.IUpdateAlgorithm;
import org.deeplearning4j.rl4j.agent.learning.algorithm.dqn.BaseTransitionTDAlgorithm;
import org.deeplearning4j.rl4j.agent.learning.algorithm.dqn.DoubleDQN;
import org.deeplearning4j.rl4j.agent.learning.algorithm.dqn.StandardDQN;
import org.deeplearning4j.rl4j.agent.learning.behavior.ILearningBehavior;
import org.deeplearning4j.rl4j.agent.learning.behavior.LearningBehavior;
import org.deeplearning4j.rl4j.agent.learning.update.FeaturesLabels;
import org.deeplearning4j.rl4j.agent.learning.update.IUpdateRule;
import org.deeplearning4j.rl4j.agent.learning.update.UpdateRule;
import org.deeplearning4j.rl4j.agent.learning.update.updater.INeuralNetUpdater;
import org.deeplearning4j.rl4j.agent.learning.update.updater.NeuralNetUpdaterConfiguration;
import org.deeplearning4j.rl4j.agent.learning.update.updater.sync.SyncLabelsNeuralNetUpdater;
import org.deeplearning4j.rl4j.experience.ExperienceHandler;
import org.deeplearning4j.rl4j.experience.ReplayMemoryExperienceHandler;
import org.deeplearning4j.rl4j.learning.IHistoryProcessor;
import org.deeplearning4j.rl4j.learning.Learning;
import org.deeplearning4j.rl4j.learning.configuration.QLearningConfiguration;
import org.deeplearning4j.rl4j.experience.StateActionRewardState;
import org.deeplearning4j.rl4j.learning.sync.qlearning.QLearning;
import org.deeplearning4j.rl4j.mdp.MDP;
import org.deeplearning4j.rl4j.network.CommonOutputNames;
import org.deeplearning4j.rl4j.network.ITrainableNeuralNet;
import org.deeplearning4j.rl4j.network.dqn.IDQN;
import org.deeplearning4j.rl4j.observation.Observation;
import org.deeplearning4j.rl4j.policy.DQNPolicy;
import org.deeplearning4j.rl4j.policy.EpsGreedy;
import org.deeplearning4j.rl4j.space.DiscreteSpace;
import org.deeplearning4j.rl4j.space.Encodable;
import org.deeplearning4j.rl4j.util.LegacyMDPWrapper;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.rng.Random;
import org.nd4j.linalg.factory.Nd4j;


public abstract class QLearningDiscrete<O extends Encodable> extends QLearning<O, Integer, DiscreteSpace> {

    @Getter
    final private QLearningConfiguration configuration;
    private final LegacyMDPWrapper<O, Integer, DiscreteSpace> mdp;
    @Getter
    private DQNPolicy<O> policy;
    @Getter
    private EpsGreedy<Integer> egPolicy;

    @Getter
    final private IDQN qNetwork;

    private int lastAction;
    private double accuReward = 0;

    private final ILearningBehavior<Integer> learningBehavior;

    protected LegacyMDPWrapper<O, Integer, DiscreteSpace> getLegacyMDPWrapper() {
        return mdp;
    }

    public QLearningDiscrete(MDP<O, Integer, DiscreteSpace> mdp, IDQN dqn, QLearningConfiguration conf, int epsilonNbStep) {
        this(mdp, dqn, conf, epsilonNbStep, Nd4j.getRandomFactory().getNewRandomInstance(conf.getSeed()));
    }

    public QLearningDiscrete(MDP<O, Integer, DiscreteSpace> mdp, IDQN dqn, QLearningConfiguration conf, int epsilonNbStep, Random random) {
        this(mdp, dqn, conf, epsilonNbStep, buildLearningBehavior(dqn, conf, random), random);
    }

    public QLearningDiscrete(MDP<O, Integer, DiscreteSpace> mdp, IDQN dqn, QLearningConfiguration conf,
                             int epsilonNbStep, ILearningBehavior<Integer> learningBehavior, Random random) {
        this.configuration = conf;
        this.mdp = new LegacyMDPWrapper<>(mdp, null);
        qNetwork = dqn;
        policy = new DQNPolicy(getQNetwork());
        egPolicy = new EpsGreedy(policy, mdp, conf.getUpdateStart(), epsilonNbStep, random, conf.getMinEpsilon(),
                this);

        this.learningBehavior = learningBehavior;
    }

    private static ILearningBehavior<Integer> buildLearningBehavior(IDQN qNetwork, QLearningConfiguration conf, Random random) {
        ITrainableNeuralNet target = qNetwork.clone();
        BaseTransitionTDAlgorithm.Configuration aglorithmConfiguration = BaseTransitionTDAlgorithm.Configuration.builder()
                .gamma(conf.getGamma())
                .errorClamp(conf.getErrorClamp())
                .build();
        IUpdateAlgorithm<FeaturesLabels, StateActionRewardState<Integer>> updateAlgorithm = conf.isDoubleDQN()
            ? new DoubleDQN(qNetwork, target, aglorithmConfiguration)
            : new StandardDQN(qNetwork, target, aglorithmConfiguration);

        NeuralNetUpdaterConfiguration neuralNetUpdaterConfiguration = NeuralNetUpdaterConfiguration.builder()
            .targetUpdateFrequency(conf.getTargetDqnUpdateFreq())
                .build();
        INeuralNetUpdater<FeaturesLabels> updater = new SyncLabelsNeuralNetUpdater(qNetwork, target, neuralNetUpdaterConfiguration);
        IUpdateRule<StateActionRewardState<Integer>> updateRule = new UpdateRule<FeaturesLabels, StateActionRewardState<Integer>>(updateAlgorithm, updater);

        ReplayMemoryExperienceHandler.Configuration experienceHandlerConfiguration = ReplayMemoryExperienceHandler.Configuration.builder()
            .maxReplayMemorySize(conf.getExpRepMaxSize())
            .batchSize(conf.getBatchSize())
            .build();
        ExperienceHandler<Integer, StateActionRewardState<Integer>> experienceHandler = new ReplayMemoryExperienceHandler(experienceHandlerConfiguration, random);
        return LearningBehavior.<Integer, StateActionRewardState<Integer>>builder()
                .experienceHandler(experienceHandler)
                .updateRule(updateRule)
                .build();
    }

    public MDP<O, Integer, DiscreteSpace> getMdp() {
        return mdp.getWrappedMDP();
    }

    public void postEpoch() {

        if (getHistoryProcessor() != null)
            getHistoryProcessor().stopMonitor();

    }

    public void preEpoch() {
        lastAction = mdp.getActionSpace().noOp();
        accuReward = 0;
        learningBehavior.handleEpisodeStart();
    }

    @Override
    public void setHistoryProcessor(IHistoryProcessor historyProcessor) {
        super.setHistoryProcessor(historyProcessor);
        mdp.setHistoryProcessor(historyProcessor);
    }

    /**
     * Single step of training
     *
     * @param obs last obs
     * @return relevant info for next step
     */
    protected QLStepReturn<Observation> trainStep(Observation obs) {

        Double maxQ = Double.NaN; //ignore if Nan for stats

        //if step of training, just repeat lastAction
        if (!obs.isSkipped()) {
            INDArray qs = getQNetwork().output(obs).get(CommonOutputNames.QValues);
            int maxAction = Learning.getMaxAction(qs);
            maxQ = qs.getDouble(maxAction);

            lastAction = getEgPolicy().nextAction(obs);
        }

        StepReply<Observation> stepReply = mdp.step(lastAction);
        accuReward += stepReply.getReward() * configuration.getRewardFactor();

        //if it's not a skipped frame, you can do a step of training
        if (!obs.isSkipped()) {

            // Add experience
            learningBehavior.handleNewExperience(obs, lastAction, accuReward, stepReply.isDone());
            accuReward = 0;
        }

        return new QLStepReturn<>(maxQ, getQNetwork().getLatestScore(), stepReply);
    }

    @Override
    protected void finishEpoch(Observation observation) {
        learningBehavior.handleEpisodeEnd(observation);
    }
}
