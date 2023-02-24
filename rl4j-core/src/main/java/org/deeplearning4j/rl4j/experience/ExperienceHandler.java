/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */
package org.deeplearning4j.rl4j.experience;

import org.deeplearning4j.rl4j.observation.Observation;

import java.util.List;

public interface ExperienceHandler<A, E> {
    void addExperience(Observation observation, A action, double reward, boolean isTerminal);

    /**
     * Called when the episode is done with the last observation
     * @param observation
     */
    void setFinalObservation(Observation observation);

    /**
     * @return The size of the list that will be returned by generateTrainingBatch().
     */
    int getTrainingBatchSize();

    /**
     * @return True if a batch is ready for training.
     */
    boolean isTrainingBatchReady();

    /**
     * The elements are returned in the historical order (i.e. in the order they happened)
     * @return The list of experience elements
     */
    List<E> generateTrainingBatch();

    /**
     * Signal the experience handler that a new episode is starting
     */
    void reset();
}
