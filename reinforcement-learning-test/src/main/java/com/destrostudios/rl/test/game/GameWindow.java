package com.destrostudios.rl.test.game;

import javax.swing.*;
import java.awt.*;
import java.awt.event.WindowAdapter;
import java.awt.event.WindowEvent;

public class GameWindow extends Frame {

    public GameWindow(FlappyBird flappyBird) {
        this.flappyBird = flappyBird;
        setTitle(Constant.GAME_TITLE);
        setLocation(Constant.FRAME_X, Constant.FRAME_Y);
        setSize(Constant.FRAME_WIDTH, Constant.FRAME_HEIGHT);
        setResizable(false);
        setVisible(true);
        addWindowListener(new WindowAdapter() {

            @Override
            public void windowClosing(WindowEvent evt) {
                System.exit(0);
            }
        });
        new Thread(() -> {
            while (true) {
                try {
                    Thread.sleep(Constant.FPS);
                } catch (InterruptedException ex) {
                    throw new RuntimeException(ex);
                }
                SwingUtilities.invokeLater(this::repaint);
            }
        }).start();
    }
    private FlappyBird flappyBird;

    @Override
    public void update(Graphics graphics) {
        graphics.drawImage(flappyBird.getImage(), 0, 0, null);
    }
}
