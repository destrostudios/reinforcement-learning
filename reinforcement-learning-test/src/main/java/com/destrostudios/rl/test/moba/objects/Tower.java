package com.destrostudios.rl.test.moba.objects;

import com.destrostudios.rl.test.moba.MobaObject;

public class Tower extends MobaObject {

    public Tower(int team, int x, int y) {
        super(team, x, y, 30, 2, 2, 4);
    }

    @Override
    public void nextFrame() {
        super.nextFrame();
        MobaObject target = getPossibleAttackTarget();
        if (target != null) {
            attack(target);
        }
    }

    @Override
    protected String getAsciiLetter() {
        return "T";
    }
}
