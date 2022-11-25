package com.destrostudios.rl.training;

import ai.djl.Model;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.training.tracker.LinearTracker;
import com.destrostudios.rl.*;
import com.destrostudios.rl.agents.EpsilonGreedyAgent;
import com.destrostudios.rl.agents.QAgent;
import com.destrostudios.rl.buffers.LruReplayBuffer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.concurrent.*;

public class Trainer {

    private static final Logger logger = LoggerFactory.getLogger(Trainer.class);

    public Trainer(Environment environment, TrainerConfig config) {
        this.environment = environment;
        this.config = config;
        this.replayBuffer = new LruReplayBuffer(config.getReplayBatchSize(), config.getReplayBufferSize());
        baseManager = NDManager.newBaseManager();
    }
    private Environment environment;
    private TrainerConfig config;
    private NDManager baseManager;
    private int environmentStep;
    private int trainStep;
    private ReplayBuffer replayBuffer;

    public void train(Model model) {
        environment.initialize(baseManager);
        try (ai.djl.training.Trainer trainer = model.newTrainer(config.getTrainingConfig())) {
            LinkedList<Long> shape = new LinkedList<>();
            shape.add((long) config.getReplayBatchSize());
            for (long shapeValue : config.getShape()) {
                shape.add(shapeValue);
            }
            trainer.initialize(new Shape(shape));
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
                agent.train(replayBuffer.getTrainingBatch());
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
            NDList preObservation = environment.getCurrentObservation();
            float reward = environment.takeAction(action);
            NDList postObservation = environment.getCurrentObservation();
            NDManager subManager = baseManager.newSubManager();
            Replay replay = new Replay(subManager, preObservation, action, postObservation, subManager.create(reward), environment.isTerminated());
            replayBuffer.addReplay(replay);
            environmentStep++;
            logger.info("ENVIRONMENT_STEP " + environmentStep);
            if ((environmentStep % 5000) == 0) {
                replayBuffer.cleanupInterval();
            }
        }
    }
}
