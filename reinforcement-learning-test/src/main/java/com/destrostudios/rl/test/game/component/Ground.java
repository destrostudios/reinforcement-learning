package com.destrostudios.rl.test.game.component;

import java.awt.Graphics;
import java.awt.image.BufferedImage;

import com.destrostudios.rl.test.game.Constant;
import com.destrostudios.rl.test.game.GameUtil;

public class Ground {

	private static final BufferedImage backgroundImage;
	public static final int GROUND_HEIGHT;
	static {
		backgroundImage = GameUtil.loadBufferedImage(Constant.BG_IMG_PATH);
		GROUND_HEIGHT = backgroundImage.getHeight();
	}

	public Ground() {
		this.velocity = Constant.GAME_SPEED;
		this.layerX = 0;
	}
	private int velocity;
	private int layerX;

	public void draw(Graphics g, Bird bird) {
		if (bird.isDead()) {
			return;
		}
		int imgWidth = backgroundImage.getWidth();
		int count = Constant.FRAME_WIDTH / imgWidth + 2; // 根据窗口宽度得到图片的绘制次数
		for (int i = 0; i < count; i++) {
			g.drawImage(backgroundImage, imgWidth * i - layerX, Constant.FRAME_HEIGHT - GROUND_HEIGHT, null);
		}
		movement();
	}

	private void movement() {
		layerX += velocity;
		if (layerX > backgroundImage.getWidth()) {
			layerX = 0;
		}
	}
}
