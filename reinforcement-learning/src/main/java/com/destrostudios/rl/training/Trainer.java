package com.destrostudios.rl.training;

import ai.djl.Model;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.types.Shape;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.TrainingConfig;
import ai.djl.training.evaluator.Accuracy;
import ai.djl.training.initializer.NormalInitializer;
import ai.djl.training.listener.TrainingListener;
import ai.djl.training.loss.Loss;
import ai.djl.training.optimizer.Adam;
import ai.djl.training.tracker.LinearTracker;
import ai.djl.training.tracker.Tracker;
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

    public static final int OBSERVE = 1000; // game steps to observe before training
    public static final int EXPLORE = 3000000; // train steps over which to anneal epsilon
    public static final int REPLAY_BUFFER_SIZE = 50000; // number of previous transitions to remember
    public static final float REWARD_DISCOUNT = 0.9f; // decay rate of past observations
    public static final float INITIAL_EPSILON = 0.01f;
    public static final float FINAL_EPSILON = 0.0001f;
    public static final int SAVE_EVERY_TRAIN_STEPS = 100000; // save model every x train steps

    public Trainer(Environment environment, TrainingConfig config, int batchSize) {
        this.environment = environment;
        this.config = config;
        this.batchSize = batchSize;
        this.outcomeBuffer = new LruOutcomeBuffer(batchSize, REPLAY_BUFFER_SIZE);
    }
    private Environment environment;
    private TrainingConfig config;
    private int batchSize;
    private int trainStep;
    private int environmentStep;
    private OutcomeBuffer outcomeBuffer;

    public static DefaultTrainingConfig createDefaultConfig() {
        return new DefaultTrainingConfig(Loss.l2Loss())
                .optOptimizer(Adam.builder().optLearningRateTracker(Tracker.fixed(1e-6f)).build())
                .addEvaluator(new Accuracy())
                .optInitializer(new NormalInitializer())
                .addTrainingListeners(TrainingListener.Defaults.basic());
    }

    public void train(Model model) {
        try (ai.djl.training.Trainer trainer = model.newTrainer(config)) {
            trainer.initialize(new Shape(batchSize, 4, 80, 80));
            trainer.notifyListeners(listener -> listener.onTrainingBegin(trainer));

            EpsilonGreedyAgent agent = new EpsilonGreedyAgent(
                new QAgent(trainer, REWARD_DISCOUNT),
                new LinearTracker.Builder()
                    .setBaseValue(INITIAL_EPSILON)
                    .optSlope(-(INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE)
                    .optMinValue(FINAL_EPSILON)
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
        while (trainStep < Trainer.EXPLORE) {
            // Needed to ensure the change in environmentStep is noticed (multithreading)
            try {
                Thread.sleep(0);
            } catch (InterruptedException ex) {
                throw new RuntimeException(ex);
            }
            if (environmentStep > Trainer.OBSERVE) {
                agent.train(outcomeBuffer.getTrainingBatch());
                trainStep++;
                logger.info("TRAIN_STEP " + trainStep);
                if ((trainStep % Trainer.SAVE_EVERY_TRAIN_STEPS) == 0) {
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
        while (trainStep < Trainer.EXPLORE) {
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
