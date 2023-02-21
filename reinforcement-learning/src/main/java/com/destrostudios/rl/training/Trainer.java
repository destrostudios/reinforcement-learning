package com.destrostudios.rl.training;

import ai.djl.Model;
import ai.djl.modality.rl.agent.EpsilonGreedy;
import ai.djl.modality.rl.agent.QAgent;
import ai.djl.modality.rl.env.RlEnv;
import ai.djl.ndarray.NDManager;
import ai.djl.training.tracker.LinearTracker;
import com.destrostudios.rl.Environment;

import java.io.IOException;
import java.nio.file.Paths;

public class Trainer {

    public Trainer(Environment environment, TrainerConfig config) {
        baseManager = NDManager.newBaseManager();
        env = new TrainerEnvironment(baseManager, environment, config);
        this.config = config;
    }
    private NDManager baseManager;
    private TrainerEnvironment env;
    private TrainerConfig config;
    private int trainStep;

    public void train(Model model) {
        try (ai.djl.training.Trainer trainer = model.newTrainer(config.getTrainingConfig())) {
            trainer.initialize(config.getShapes());

            trainer.notifyListeners(listener -> listener.onTrainingBegin(trainer));

            EpsilonGreedy agent = new EpsilonGreedy(
                new QAgent(trainer, config.getRewardDiscount()),
                LinearTracker.builder()
                    .setBaseValue(config.getInitialEpsilon())
                    .optSlope((config.getFinalEpsilon() - config.getInitialEpsilon()) / config.getEnvironmentStepsExplore())
                    .optMaxUpdates(config.getEnvironmentStepsExplore())
                    .build()
            );

            for (int i = 0; i < config.getTrainStepsTotal(); i++) {
                for (int r = 0; r < 100; r++) {
                    NDManager trainSubManager = baseManager.newSubManager();
                    env.setTrainSubManager(trainSubManager);
                    env.runEnvironment(agent, true);
                    trainSubManager.close();
                    RlEnv.Step[] steps = env.getBatch();
                    agent.trainBatch(steps);
                    trainer.step();
                    trainStep++;
                    if ((trainStep % config.getTrainStepsSaveInterval()) == 0) {
                        try {
                            model.save(Paths.get("."), "dqn-" + trainStep);
                        } catch (IOException ex) {
                            throw new RuntimeException(ex);
                        }
                    }
                }
                trainer.notifyListeners(listener -> listener.onEpoch(trainer));
            }
        }
    }
}
