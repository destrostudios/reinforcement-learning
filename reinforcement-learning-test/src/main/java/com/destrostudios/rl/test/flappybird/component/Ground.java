package com.destrostudios.rl.test.flappybird.component;

import java.awt.Graphics;
import java.awt.image.BufferedImage;

import com.destrostudios.rl.test.flappybird.Constant;
import com.destrostudios.rl.test.flappybird.GameUtil;

public class Ground {

	private static final BufferedImage backgroundImage;
	public static final int GROUND_HEIGHT;
	static {
		backgroundImage = GameUtil.loadBufferedImage(Constant.BACKGROUND_IMAGE_PATH);
		GROUND_HEIGHT = backgroundImage.getHeight();
	}

	public Ground() {
		this.velocity = Constant.GAME_SPEED;
		this.layerX = 0;
	}
	private int velocity;
	private int layerX;

	public void update() {
		layerX += velocity;
		if (layerX > backgroundImage.getWidth()) {
			layerX = 0;
		}
	}

	public void draw(Graphics graphics) {
		int imgWidth = backgroundImage.getWidth();
		int count = Constant.FRAME_WIDTH / imgWidth + 2;
		for (int i = 0; i < count; i++) {
			graphics.drawImage(backgroundImage, imgWidth * i - layerX, Constant.FRAME_HEIGHT - GROUND_HEIGHT, null);
		}
	}
}
