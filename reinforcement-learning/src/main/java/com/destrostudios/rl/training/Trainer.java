package com.destrostudios.rl.training;

import ai.djl.Model;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.types.Shape;
import ai.djl.training.tracker.LinearTracker;
import com.destrostudios.rl.Outcome;
import com.destrostudios.rl.OutcomeBuffer;
import com.destrostudios.rl.agents.EpsilonGreedyAgent;
import com.destrostudios.rl.agents.QAgent;
import com.destrostudios.rl.Agent;
import com.destrostudios.rl.Environment;
import com.destrostudios.rl.buffers.LruOutcomeBuffer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.concurrent.*;

public class Trainer {

    private static final Logger logger = LoggerFactory.getLogger(Trainer.class);

    public Trainer(Environment environment, TrainerConfig config) {
        this.environment = environment;
        this.config = config;
        this.outcomeBuffer = new LruOutcomeBuffer(config.getOutcomeBatchSize(), config.getOutcomeBufferSize());
    }
    private Environment environment;
    private TrainerConfig config;
    private int trainStep;
    private int environmentStep;
    private OutcomeBuffer outcomeBuffer;

    public void train(Model model) {
        try (ai.djl.training.Trainer trainer = model.newTrainer(config.getTrainingConfig())) {
            trainer.initialize(new Shape(config.getOutcomeBatchSize(), 4, 80, 80));
            trainer.notifyListeners(listener -> listener.onTrainingBegin(trainer));

            EpsilonGreedyAgent agent = new EpsilonGreedyAgent(
                new QAgent(trainer, config.getRewardDiscount()),
                new LinearTracker.Builder()
                    .setBaseValue(config.getInitialEpsilon())
                    .optSlope(-(config.getInitialEpsilon() - config.getFinalEpsilon()) / config.getTrainStepsExplore())
                    .optMinValue(config.getFinalEpsilon())
                    .build()
            );

            ExecutorService executorService = Executors.newFixedThreadPool(2);
            ArrayList<Runnable> runnables = new ArrayList<>();
            runnables.add(() -> trainLoop(agent, model));
            runnables.add(() -> runLoop(agent));
            try {
                ArrayList<Future<?>> futures = new ArrayList<>();
                for (Runnable runnable : runnables) {
                    futures.add(executorService.submit(runnable));
                }
                for (Future<?> future : futures) {
                    future.get();
                }
            } catch (InterruptedException | ExecutionException ex) {
                throw new RuntimeException(ex);
            } finally {
                executorService.shutdown();
            }
        }
    }

    private void trainLoop(Agent agent, Model model) {
        while (trainStep < config.getTrainStepsExplore()) {
            // Needed to ensure the change in environmentStep is noticed (multithreading)
            try {
                Thread.sleep(0);
            } catch (InterruptedException ex) {
                throw new RuntimeException(ex);
            }
            if (environmentStep > config.getEnvironmentStepsObserve()) {
                agent.train(outcomeBuffer.getTrainingBatch());
                trainStep++;
                logger.info("TRAIN_STEP " + trainStep);
                if ((trainStep % config.getTrainStepsSaveInterval()) == 0) {
                    try {
                        model.save(Paths.get("."), "dqn-" + trainStep);
                    } catch (IOException ex) {
                        throw new RuntimeException(ex);
                    }
                }
            }
        }
    }

    private void runLoop(Agent agent) {
        while (trainStep < config.getTrainStepsExplore()) {
            NDList action = agent.chooseAction(environment, true);
            Outcome step = environment.takeAction(action);
            outcomeBuffer.addOutcome(step);
            environmentStep++;
            logger.info("ENVIRONMENT_STEP " + environmentStep);
            if ((environmentStep % 5000) == 0) {
                outcomeBuffer.cleanupInterval();
            }
        }
    }
}
