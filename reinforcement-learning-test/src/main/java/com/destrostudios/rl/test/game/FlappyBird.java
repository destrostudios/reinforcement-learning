package com.destrostudios.rl.test.game;

import com.destrostudios.rl.EnvironmentStep;
import com.destrostudios.rl.test.game.component.Ground;
import com.destrostudios.rl.test.game.component.Bird;
import com.destrostudios.rl.test.game.component.GameElementLayer;
import com.destrostudios.rl.test.game.component.ScoreCounter;
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

    public FlappyBird(NDManager manager, boolean withGraphics) {
        this.manager = manager;
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
    @Setter
    private float currentReward;
    @Setter
    private boolean currentTerminal;
    @Setter
    private int gameState;
    private Ground ground;
    private Bird bird;
    private GameElementLayer gameElement;
    private boolean withGraphics;
    private NDManager manager;
    private BufferedImage currentImage;
    private NDList currentObservation;
    @Getter
    private ArrayList<NDList> actionSpace;
    private Queue<NDArray> imgQueue = new ArrayDeque<>(4);
    @Getter
    private ScoreCounter scoreCounter;

    @Override
    public EnvironmentStep runEnvironment(Agent agent, boolean training) {
        currentTerminal = false;
        currentReward = 0.2f;

        NDList action = agent.chooseAction(this, training);

        // action[0] == 1 : do nothing
        // action[1] == 1 : flap the bird
        if (action.singletonOrThrow().getInt(1) == 1) {
            bird.birdFlap();
        }

        bird.update();
        ground.update(bird);
        gameElement.update(bird);

        NDList preObservation = currentObservation;
        currentObservation = createObservation(currentImage);

        FlappyBirdStep step = new FlappyBirdStep(manager.newSubManager(), preObservation, currentObservation, action, currentReward, currentTerminal);
        logger.info( "ACTION " + Arrays.toString(action.singletonOrThrow().toArray()) + " / REWARD " + step.getReward().getFloat() + " / SCORE " + scoreCounter.getScore());

        if (gameState == GAME_OVER) {
            restartGame();
        }

        drawImage();
        if (withGraphics) {
            repaint();
            try {
                Thread.sleep(Constant.FPS);
            } catch (InterruptedException ex) {
                ex.printStackTrace();
            }
        }

        return step;
    }

    @Override
    public NDList getObservation() {
        return currentObservation;
    }

    @Override
    public void close() {
        manager.close();
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

    private void drawImage() {
        Graphics graphics = currentImage.getGraphics();
        graphics.setColor(Constant.BG_COLOR);
        graphics.fillRect(0, 0, Constant.FRAME_WIDTH, Constant.FRAME_HEIGHT);
        ground.draw(graphics);
        bird.draw(graphics);
        gameElement.draw(graphics);
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
}
