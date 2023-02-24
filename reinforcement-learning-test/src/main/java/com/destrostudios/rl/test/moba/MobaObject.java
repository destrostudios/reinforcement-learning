package com.destrostudios.rl.test.moba;

import lombok.Getter;
import lombok.Setter;

import java.util.List;
import java.util.stream.Collectors;

@Getter
public abstract class MobaObject {

    public MobaObject(int team, int x, int y, int maximumHealth, int attackDamage, int attackRange, int attackCooldown) {
        this.team = team;
        this.x = x;
        this.y = y;
        this.maximumHealth = maximumHealth;
        this.attackDamage = attackDamage;
        this.attackRange = attackRange;
        this.attackCooldown = attackCooldown;
        setToMaximumHealth();
    }
    @Setter
    protected MobaMap map;
    @Getter
    protected int team;
    @Getter
    protected int x;
    @Getter
    protected int y;
    @Getter
    protected int maximumHealth;
    @Getter
    protected int health;
    @Getter
    protected int attackDamage;
    @Getter
    protected int attackRange;
    @Getter
    protected int attackCooldown;
    @Getter
    protected int remainingAttackCooldown;

    public void nextFrame() {
        if (remainingAttackCooldown > 0) {
            remainingAttackCooldown--;
        }
    }

    public void moveLeft() {
        tryMove(x - 1, y);
    }

    public void moveRight() {
        tryMove(x + 1, y);
    }

    public void moveUp() {
        tryMove(x, y - 1);
    }

    public void moveDown() {
        tryMove(x, y + 1);
    }

    public void attack(int targetX, int targetY) {
        MobaObject target = map.getObject(targetX, targetY);
        if (target != null) {
            attack(target);
        }
    }

    public void attack(MobaObject target) {
        if (canAttack(target)) {
            target.damage(attackDamage);
            remainingAttackCooldown = attackCooldown;
        }
    }

    public void damage(int damage) {
        health -= damage;
        if (isDead()) {
            onDeath();
        }
    }

    protected void onDeath() {

    }

    public void setToMaximumHealth() {
        health = maximumHealth;
    }

    public boolean isDead() {
        return (health <= 0);
    }

    private void tryMove(int targetX, int targetY) {
        if (map.isFree(targetX, targetY)) {
            x = targetX;
            y = targetY;
        }
    }

    public MobaObject getPossibleAttackTarget() {
        return map.getObjects().stream().filter(this::canAttack).findFirst().orElse(null);
    }

    public boolean canAttack(MobaObject object) {
        if (remainingAttackCooldown > 0) {
            return false;
        }
        return (object.getTeam() != team) && (getDistance(object) <= attackRange);
    }

    public List<MobaObject> getNearestObjects(int limit) {
        return map.getObjects().stream()
                .filter(object -> object != this)
                .limit(limit)
                .collect(Collectors.toList());
    }

    public int getDistance(MobaObject target) {
        // Good enough approximation
        return Math.max(getDistanceX(target.getX()), getDistanceY(target.getY()));
    }

    public int getDistanceX(int targetX) {
        return getDistanceOnAxis(x, targetX);
    }

    public int getDistanceY(int targetY) {
        return getDistanceOnAxis(y, targetY);
    }

    private static int getDistanceOnAxis(int source, int target) {
        return Math.abs(source - target);
    }

    public String getAsciiImage() {
        return getAsciiLetter() + ((team == 1) ? "1" : "2") + "(" + (((health > 0) && (health < 10)) ? "0" : "") + health + ")";
    }

    protected abstract String getAsciiLetter();
}
