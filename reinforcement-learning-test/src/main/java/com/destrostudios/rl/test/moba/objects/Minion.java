package com.destrostudios.rl.test.moba.objects;

import com.destrostudios.rl.test.moba.MobaObject;

public class Minion extends MobaObject {

    public Minion(int team, int x, int y) {
        super(team, x, y, 5, 1, 1, 2);
    }

    @Override
    public void nextFrame() {
        super.nextFrame();
        MobaObject target = getPossibleAttackTarget();
        if (target != null) {
            attack(target);
        } else {
            if (team == 1) {
                moveRight();
            } else {
                moveLeft();
            }
        }
    }

    @Override
    protected void onDeath() {
        super.onDeath();
        map.remove(this);
    }

    @Override
    protected String getAsciiLetter() {
        return "M";
    }
}
