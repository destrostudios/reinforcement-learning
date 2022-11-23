package com.destrostudios.rl.test.game;

import com.destrostudios.rl.EnvironmentStep;
import com.destrostudios.rl.test.game.component.Ground;
import com.destrostudios.rl.test.game.component.Bird;
import com.destrostudios.rl.test.game.component.GameElementLayer;
import com.destrostudios.rl.test.game.component.ScoreCounter;
import com.destrostudios.rl.buffers.LruReplayBuffer;
import com.destrostudios.rl.ReplayBuffer;
import com.destrostudios.rl.Agent;
import com.destrostudios.rl.Environment;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import lombok.Getter;
import lombok.Setter;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.awt.*;
import java.awt.event.WindowAdapter;
import java.awt.event.WindowEvent;
import java.awt.image.BufferedImage;
import java.util.*;

public class FlappyBird extends Frame implements Environment {

    private static final Logger logger = LoggerFactory.getLogger(FlappyBird.class);
    public static final int GAME_START = 1;
    public static final int GAME_OVER = 2;

    /**
     * Constructs a {@link FlappyBird} with a basic {@link LruReplayBuffer}.
     *
     * @param manager          the manager for creating the game
     * @param batchSize        the number of steps to train on per batch
     * @param replayBufferSize the number of steps to hold in the buffer
     */
    public FlappyBird(NDManager manager, int batchSize, int replayBufferSize, boolean withGraphics) {
        this.manager = manager;
        this.replayBuffer = new LruReplayBuffer(batchSize, replayBufferSize);
        this.withGraphics = withGraphics;
        if (this.withGraphics) {
            initFrame();
            this.setVisible(true);
        }
        actionSpace = new ArrayList<>();
        actionSpace.add(new NDList(manager.create(Constant.DO_NOTHING)));
        actionSpace.add(new NDList(manager.create(Constant.FLAP)));

        currentImage = new BufferedImage(Constant.FRAME_WIDTH, Constant.FRAME_HEIGHT, BufferedImage.TYPE_4BYTE_ABGR);
        currentObservation = createObservation(currentImage);
        ground = new Ground();
        gameElement = new GameElementLayer();
        bird = new Bird(this);
        gameState = GAME_START;

        scoreCounter = new ScoreCounter(this);
    }
    @Getter
    private int environmentStep = 0;
    @Setter
    private boolean currentTerminal = false;
    @Setter
    private float currentReward = 0.2f;
    @Setter
    private int gameState;
    private Ground ground;
    private Bird bird;
    private GameElementLayer gameElement;
    private boolean withGraphics;
    private NDManager manager;
    private ReplayBuffer replayBuffer;
    private BufferedImage currentImage;
    private NDList currentObservation;
    @Getter
    private ArrayList<NDList> actionSpace;
    private Queue<NDArray> imgQueue = new ArrayDeque<>(4);
    @Getter
    private ScoreCounter scoreCounter;

    @Override
    public EnvironmentStep[] runEnvironment(Agent agent, boolean training) {
        EnvironmentStep[] batchSteps = new EnvironmentStep[0];
        reset();

        // run the game
        NDList action = agent.chooseAction(this, training);
        step(action, training);
        if (training) {
            batchSteps = replayBuffer.getBatch();
        }
        if ((environmentStep % 5000) == 0) {
            closeStep();
        }
        environmentStep++;
        return batchSteps;
    }

    /**
     * action[0] == 1 : do nothing
     * action[1] == 1 : flap the bird
     */
    private void step(NDList action, boolean training) {
        if (action.singletonOrThrow().getInt(1) == 1) {
            bird.birdFlap();
        }
        if (withGraphics) {
            drawEnvironment();
        }

        NDList preObservation = currentObservation;
        currentObservation = createObservation(currentImage);

        FlappyBirdStep step = new FlappyBirdStep(manager.newSubManager(), preObservation, currentObservation, action, currentReward, currentTerminal);
        if (training) {
            replayBuffer.addStep(step);
        }
        logger.info(
            "ENVIRONMENT_STEP " + environmentStep +
            " / " + "ACTION " + (Arrays.toString(action.singletonOrThrow().toArray())) +
            " / " + "REWARD " + step.getReward().getFloat() +
            " / " + "SCORE " + getScore()
        );
        if (gameState == GAME_OVER) {
            restartGame();
        }
    }

    @Override
    public NDList getObservation() {
        return currentObservation;
    }

    /**
     * Close the steps in replayBuffer which are not pointed to.
     */
    public void closeStep() {
        replayBuffer.closeStep();
    }

    @Override
    public void close() {
        manager.close();
    }

    @Override
    public void reset() {
        currentReward = 0.2f;
        currentTerminal = false;
    }

    /**
     * Convert image to CNN input.
     * Copy the initial frame image, stack into NDList, then replace the fourth frame with the current frame to ensure that the batch picture is continuous.
     *
     * @return the CNN input
     */
    public NDList createObservation(BufferedImage image) {
        NDArray observation = GameUtil.preprocessImage(image, 80, 80);
        if (imgQueue.isEmpty()) {
            for (int i = 0; i < 4; i++) {
                imgQueue.offer(observation);
            }
            return new NDList(NDArrays.stack(new NDList(observation, observation, observation, observation), 1));
        } else {
            imgQueue.remove();
            imgQueue.offer(observation);
            NDArray[] buf = new NDArray[4];
            int i = 0;
            for (NDArray nd : imgQueue) {
                buf[i++] = nd;
            }
            return new NDList(NDArrays.stack(new NDList(buf[0], buf[1], buf[2], buf[3]), 1));
        }
    }

    private void drawEnvironment() {
        Graphics graphics = currentImage.getGraphics();
        graphics.setColor(Constant.BG_COLOR);
        graphics.fillRect(0, 0, Constant.FRAME_WIDTH, Constant.FRAME_HEIGHT);
        ground.draw(graphics, bird);
        bird.draw(graphics);
        gameElement.draw(graphics, bird);
        repaint();
        try {
            Thread.sleep(Constant.FPS);
        } catch (InterruptedException ex) {
            ex.printStackTrace();
        }
    }

    private void initFrame() {
        setSize(Constant.FRAME_WIDTH, Constant.FRAME_HEIGHT);
        setTitle(Constant.GAME_TITLE);
        setLocation(Constant.FRAME_X, Constant.FRAME_Y);
        setResizable(false);
        setVisible(true);
        addWindowListener(new WindowAdapter() {
            @Override
            public void windowClosing(WindowEvent e) {
                System.exit(0);
            }
        });
    }

    private void restartGame() {
        gameState = GAME_START;
        gameElement.reset();
        bird.reset();
    }

    @Override
    public void update(Graphics graphics) {
        graphics.drawImage(currentImage, 0, 0, null);
    }

    public long getScore() {
        return scoreCounter.getScore();
    }
}
