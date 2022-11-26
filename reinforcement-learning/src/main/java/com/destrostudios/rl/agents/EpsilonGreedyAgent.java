package com.destrostudios.rl.agents;

import com.destrostudios.rl.Agent;
import com.destrostudios.rl.Environment;
import ai.djl.ndarray.NDList;
import ai.djl.training.tracker.Tracker;
import ai.djl.util.RandomUtils;
import com.destrostudios.rl.Replay;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Simple exploration/excitation agent.
 *
 * It helps other agents explore their environments during training by sometimes picking random actions.
 *
 * If a model based agent is used, it will only explore paths through the environment that have already been seen.
 * While this is sometimes good, it is also important to sometimes explore new paths as well.
 * This agent exhibits a tradeoff that takes random paths a fixed percentage of the time during training.
 */
public class EpsilonGreedyAgent implements Agent {

    private static final Logger logger = LoggerFactory.getLogger(EpsilonGreedyAgent.class);

    public EpsilonGreedyAgent(Agent baseAgent, Tracker exploreRate) {
        this.baseAgent = baseAgent;
        this.exploreRate = exploreRate;
    }
    private Agent baseAgent;
    private Tracker exploreRate;
    private int counter;

    @Override
    public NDList chooseAction(Environment environment) {
        if (RandomUtils.random() < exploreRate.getNewValue(counter++)) {
            logger.info("***********RANDOM ACTION***********");
            return environment.getActionSpace().get(RandomUtils.nextInt(environment.getActionSpace().size()));
        }
        return baseAgent.chooseAction(environment);
    }

    @Override
    public void train(Replay[] replays) {
        baseAgent.train(replays);
    }
}
