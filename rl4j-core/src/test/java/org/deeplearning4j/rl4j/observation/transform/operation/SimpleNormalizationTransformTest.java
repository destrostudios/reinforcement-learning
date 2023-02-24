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

package org.deeplearning4j.rl4j.observation.transform.operation;

import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
@Tag(TagNames.FILE_IO)
@NativeTag
public class SimpleNormalizationTransformTest {
    @Test()
    public void when_maxIsLessThanMin_expect_exception() {
        assertThrows(IllegalArgumentException.class,() -> {
            // Arrange
            SimpleNormalizationTransform sut = new SimpleNormalizationTransform(10.0, 1.0);
        });

    }

    @Test
    public void when_transformIsCalled_expect_inputNormalized() {
        // Arrange
        SimpleNormalizationTransform sut = new SimpleNormalizationTransform(1.0, 11.0);
        INDArray input = Nd4j.create(new double[] { 1.0, 11.0 });

        // Act
        INDArray result = sut.transform(input);

        // Assert
        assertEquals(0.0, result.getDouble(0), 0.00001);
        assertEquals(1.0, result.getDouble(1), 0.00001);
    }

}
