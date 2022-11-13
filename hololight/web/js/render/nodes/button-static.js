// Copyright 2018 The Immersive Web Community Group
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

import {Material} from '../core/material.js';
import {Node} from '../core/node.js';
import {PrimitiveStream} from '../geometry/primitive-stream.js';

const BUTTON_SIZE = 0.1;
const BUTTON_CORNER_RADIUS = 0.025;
const BUTTON_CORNER_SEGMENTS = 8;
const BUTTON_LAYER_DISTANCE = 0.005;
const BUTTON_COLOR = 0;
const BUTTON_ALPHA = 1.0;
const BUTTON_HOVER_SCALE = 1.1;
const BUTTON_HOVER_TRANSITION_TIME_MS = 200;

class ButtonMaterial extends Material {
  constructor() {
    super();

    this.state.blend = true;

    this.defineUniform('hoverAmount', 0);
  }

  get materialName() {
    return 'BUTTON_MATERIAL';
  }

  get vertexSource() {
    return `
    attribute vec3 POSITION;

    uniform float hoverAmount;

    vec4 vertex_main(mat4 proj, mat4 view, mat4 model) {
      float scale = mix(1.0, ${BUTTON_HOVER_SCALE}, hoverAmount);
      vec4 pos = vec4(POSITION.x * scale, POSITION.y * scale, POSITION.z * (scale + (hoverAmount * 0.2)), 1.0);
      return proj * view * model * pos;
    }`;
  }

  get fragmentSource() {
    return `
    uniform float hoverAmount;

    const vec4 default_color = vec4(${BUTTON_COLOR+1}, ${BUTTON_COLOR}, ${BUTTON_COLOR}, ${BUTTON_ALPHA});

    vec4 fragment_main() {
      return mix(default_color, default_color, hoverAmount);
    }`;
  }
}

export class ButtonNodeIcon extends Node {
  constructor(iconTexture, callback) {
    super();
  }

  onRendererChanged(renderer) {
    let stream = new PrimitiveStream();

    let hd = BUTTON_LAYER_DISTANCE * 0.5;

    // Build a rounded rect for the background.
    let hs = BUTTON_SIZE * 0.5;
    let ihs = hs - BUTTON_CORNER_RADIUS;
    stream.startGeometry();

    // Rounded corners and sides
    let segments = BUTTON_CORNER_SEGMENTS * 4;
    for (let i = 0; i < segments; ++i) {
      let rad = i * ((Math.PI * 2.0) / segments);
      let x = Math.cos(rad) * BUTTON_CORNER_RADIUS;
      let y = Math.sin(rad) * BUTTON_CORNER_RADIUS;
      let section = Math.floor(i / BUTTON_CORNER_SEGMENTS);
      switch (section) {
        case 0:
          x += ihs;
          y += ihs;
          break;
        case 1:
          x -= ihs;
          y += ihs;
          break;
        case 2:
          x -= ihs;
          y -= ihs;
          break;
        case 3:
          x += ihs;
          y -= ihs;
          break;
      }

      stream.pushVertex(x, y, -hd, 0, 0, 0, 0, 1);

      if (i > 1) {
        stream.pushTriangle(0, i-1, i);
      }
    }

    stream.endGeometry();

    let buttonPrimitive = stream.finishPrimitive(renderer);
    this._buttonRenderPrimitive = renderer.createRenderPrimitive(buttonPrimitive, new ButtonMaterial());
    this.addRenderPrimitive(this._buttonRenderPrimitive);
  }
}
