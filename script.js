const canvas = document.querySelector("canvas");
const gl = canvas.getContext("webgl");


function cross(a, b) {
    return [
        a[1] * b[2] - a[2] * b[1],
        a[0] * b[2] - a[2] * b[0],
        a[1] * b[0] - a[0] * b[1]
    ];
}

function normalize(v) {
    const len = Math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
    return [v[0] / len, v[1] / len, v[2] / len];
}

function sub(a, b) {
    return [
        a[0] - b[0],
        a[1] - b[1],
        a[2] - b[2]
    ];
}

function gridToPositions(grid) {
    const rows = grid.length, cols = grid[0].length;
    const minx = -5, minz = -5, maxx = 5, maxz = 5;
    const stepx = (maxx - minx) / (cols - 1);
    const stepz = (maxz - minz) / (rows - 1);
    const positions = [];
    for(let row = 0;row < rows;row++) {
        for(let col = 0;col < cols;col++) {
            const x = minx + stepx * col;
            const y = grid[row][col];
            const z = minz + stepz * row;
            positions.push(x);
            positions.push(y);
            positions.push(z);
        }
    }
    return positions;
}

function positionsToNormals(positions, rows, cols) {
    const indices = makeIndexGrid(rows, cols);
    const normals = [];
    function getPosition(row, col) {
        const x = positions[indices[row][col] * 3 + 0];
        const y = positions[indices[row][col] * 3 + 1];
        const z = positions[indices[row][col] * 3 + 2];
        return [x, y, z];
    }
    for (let row = 0; row < rows; row++) {
        for (let col = 0; col < cols; col++) {
            const norms = [];
            //
            //   +---+---+
            //   |  /|  /|
            //   | / | / |
            //   |/  |/  |
            //   +---+---+
            //   |  /|  /|
            //   | / | / |
            //   |/  |/  |
            //   +---+---+
            //
            const P = getPosition(row, col);

            let W;
            let N;
            let NE;
            let E;
            let S;
            let SW;
            if (col > 0) {
                W = normalize(sub(getPosition(row, col - 1), P));
            }
            if (row > 0) {
                N = normalize(sub(getPosition(row - 1, col), P));
            }
            if (row > 0 && col < cols - 1) {
                NE = normalize(sub(getPosition(row - 1, col + 1), P));
            }
            if (col < cols - 1) {
                E = normalize(sub(getPosition(row, col + 1), P));
            }
            if (row < rows - 1) {
                S = normalize(sub(getPosition(row + 1, col), P));
            }
            if (row < rows - 1 && col > 0) {
                SW = normalize(sub(getPosition(row + 1, col - 1), P));
            }
            if (W !== undefined && N !== undefined) {
                norms.push(cross(W, N));
            }
            if (N !== undefined && NE !== undefined) {
                norms.push(cross(N, NE));
            }
            if (NE !== undefined && E !== undefined) {
                norms.push(cross(NE, E));
            }
            if (E !== undefined && S !== undefined) {
                norms.push(cross(E, S));
            }
            if (S !== undefined && SW !== undefined) {
                norms.push(cross(S, SW));
            }
            if (SW !== undefined && W !== undefined) {
                norms.push(cross(SW, W));
            }

            let norm = [0, 0, 0];
            for (const n of norms) {
                norm[0] += (n[0] / norms.length);
                norm[1] += (n[1] / norms.length);
                norm[2] += (n[2] / norms.length);
            }
            norm = normalize(norm);
            normals.push(norm[0], norm[1], norm[2]);
        }
    }
    if(normals.every((value) => Number.isNaN(value))) {
        alert("isNaN in every normal");
    }
    return normals;
}

function makeIndexGrid(rows, cols) {
    const indexGrid = [];
    let index = 0;
    for (let row = 0; row < rows; row++) {
        indexGrid.push([]);
        for (let col = 0; col < cols; col++) {
            indexGrid[row].push(index);
            index++;
        }
    }
    return indexGrid;
}

function getGridIndices(rows, cols) {
    const indexGrid = makeIndexGrid(rows, cols);
    const indices = [];
    for(let row = 0;row < rows-1;row++) {
        for(let col = 0;col < cols-1; col++) {
            indices.push(indexGrid[row][col]);
            indices.push(indexGrid[row+1][col]);
            indices.push(indexGrid[row][col+1]);

            indices.push(indexGrid[row][col+1]);
            indices.push(indexGrid[row+1][col]);
            indices.push(indexGrid[row+1][col+1]);
        }
    }
    return indices;
}

function uniform(low, hi) {
    // NewValue = (((OldValue - OldMin) * (NewMax - NewMin)) / (OldMax - OldMin)) + NewMin
    return (Math.random() * (hi - low) / (1 - 0)) + low;
}

function fixedAvg(grid, i, j, v, offsets) {
    const n = grid.length;
    let res = 0, k = 0;
    for(const [p, q] of offsets) {
        const pp = i + p * v, qq = j + q * v;
        if(0 <= pp && pp < n && 0 <= qq && qq < n) {
            res += grid[pp][qq];
            k += 1.0;
        }
    }
    return res / k;
}

function periodicAvg(grid, i, j, v, offsets) {
    const n = grid.length - 1;
    let res = 0;
    for(const [p, q] of offsets) {
        let row = (i + p * v) % n;
        if(row < 0) {
            row = grid.length + row;
        }
        let col = (j + q * v) % n;
        if(col < 0) {
            col = grid.length + col;
        }
        res += grid[row][col];
    }
    return res / 4.0;
}
// def periodic(d, i, j, v, offsets):
//     # For periodic bdries, the last row / col mirrors the first row / col.
//     # Hence the effective square size is(n - 1)x(n - 1).Redefine n accordingly!
// n = d.shape[0] - 1

// res = 0
// for p, q in offsets:
//     res += d[(i + p * v) % n, (j + q * v) % n]
// return res / 4.0

function singleDiamondSquare(grid, w, s, avg=periodicAvg) {
    const n = grid.length;
    const v = Math.floor(w / 2);
    const diamond = [[-1, -1], [-1, 1], [1, 1], [1, -1]];
    const square = [[-1, 0],[0, -1],[1, 0],[0, 1]];

    for(let i = v;i < n;i+=w) {
        for(let j = v;j < n;j+=w) {
            grid[i][j] = avg(grid, i, j, v, diamond) + uniform(-s, s);
        }
    }
    // # Square Step, rows
    // for i in range(v, n, w):
    //     for j in range(0, n, w):
    //         d[i, j] = avg(d, i, j, v, square) + random.uniform(-s, s)
    for (let i = v; i < n; i += w) {
        for (let j = 0; j < n; j += w) {
            grid[i][j] = avg(grid, i, j, v, square) + uniform(-s, s);
        }
    }
    // # Square Step, cols
    // for i in range(0, n, w):
    //     for j in range(v, n, w):
    //         d[i, j] = avg(d, i, j, v, square) + random.uniform(-s, s)
    for (let i = 0; i < n; i += w) {
        for (let j = v; j < n; j += w) {
            grid[i][j] = avg(grid, i, j, v, square) + uniform(-s, s);
        }
    }
}

function diamondSquare(n, roughness) {
    const grid = initGrid(n, n);
    let w = n;
    let s = 1.0;
    while (w > 1) {
        singleDiamondSquare(grid, w, s);
        w = Math.floor(w / 2);
        s *= roughness;
    }
    return grid;
    // w, s = n - 1, 1.0
    // while w > 1:
    //     single_diamond_square_step(d, w, s, bdry)
}

function initGrid(rows, cols) {
    const grid = [];
    for(let row = 0;row < rows;row++) {
        grid.push([]);
        for(let col = 0;col < cols;col++) {
            grid[row].push(0);
        }
    }
    return grid;
}

function initBuffers(rows, cols) {
    const positionBuffer = gl.createBuffer();
    const normalBuffer = gl.createBuffer();
    const indices = getGridIndices(rows, cols);
    const indexBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, indexBuffer);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, new Uint16Array(indices), gl.STATIC_DRAW);
    return {
        position: positionBuffer,
        index: indexBuffer,
        normal: normalBuffer,
    };
}


function initShaders() {
    const vshader = gl.createShader(gl.VERTEX_SHADER);
    gl.shaderSource(vshader, `
attribute vec3 aPos;
attribute vec3 aNormal;
uniform mat4 perspective;
uniform mat4 model;
uniform mat4 view;
varying mediump vec3 pos;
varying mediump vec3 normal;
void main() {
    gl_Position = perspective * view * model * vec4(aPos, 1);
    pos = aPos;
    normal = aNormal;
}
`);
    gl.compileShader(vshader);
    if (!gl.getShaderParameter(vshader, gl.COMPILE_STATUS)) {
        alert(`Vertex Shader compile error: ${gl.getShaderInfoLog(vshader)}`);
        return;
    }

    const fshader = gl.createShader(gl.FRAGMENT_SHADER);
    // 	(97, 54, 19)
    gl.shaderSource(fshader, `
precision mediump float;
uniform vec3 eye;
varying mediump vec3 pos;
varying mediump vec3 normal;
void main() {
    vec3 V = normalize(eye - pos);
    vec3 L = normalize(vec3(1.0, 1.0, -1.0));
    vec3 R = normalize(2.0 * dot(L, normal) * normal - L);

    vec3 color = vec3(0.0, 1.0, 0.0);
    if(dot(normal, vec3(0.0, 1.0, 0.0)) < 0.98) {
        color = vec3(0.38, 0.21, .074);
    }
    if(pos.y > 0.1) {
        color = vec3(1.0, 1.0, 1.0);
    }
    if(pos.y < -0.2) {
        color = vec3(0, 0, 1);
    }

    vec3 ambient = vec3(0.2, 0.2, 0.2);
    vec3 diffuse = max(dot(L, normal), 0.0) * color;
    vec3 spec = pow(max(dot(R, V), 0.0), 32.0) * vec3(1, 1, 1);

    gl_FragColor = vec4(ambient + diffuse + spec, 1);
}
`);
    gl.compileShader(fshader);
    if (!gl.getShaderParameter(fshader, gl.COMPILE_STATUS)) {
        alert(`fragment shader compile error: ${gl.getShaderInfoLog(fshader)}`);
        return;
    }

    const program = gl.createProgram();
    gl.attachShader(program, vshader);
    gl.attachShader(program, fshader);
    gl.linkProgram(program);
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
        alert(`program link error: ${gl.getProgramInfoLog(program)}`);
        return;
    }
    return {
        program: program,
        locations: {
            attributes: {
                aPos: gl.getAttribLocation(program, "aPos"),
                aNormal: gl.getAttribLocation(program, "aNormal"),
            },
            uniforms: {
                perspective: gl.getUniformLocation(program, "perspective"),
                model: gl.getUniformLocation(program, "model"),
                view: gl.getUniformLocation(program, "view"),
                eye: gl.getUniformLocation(program, "eye"),
            }
        }
    };
}

class Timer {
    constructor(seconds, callback) {
        this.time = 0;
        this.seconds = seconds;
        this.callback = callback;
    }
    update(deltaTime) {
        this.time += deltaTime
        if(this.time >= this.seconds) {
            this.callback();
            this.time = 0;
        }
    }
}

function medianSplit(row, start, end, displacement, roughness) {
    if(start + 1 === end) {
        return;
    }
    mid = Math.floor((start + end) / 2);
    const change = (2 * Math.random() - 1) * displacement;

    row[mid] = (row[start] + row[end]) / 2 + change;
    displacement = displacement * roughness;
    medianSplit(row, start, mid, displacement, roughness);
    medianSplit(row, mid, end, displacement, roughness);
}

class Terrain {
    constructor(k, roughness) {
        const n = Math.pow(2, k) + 1;
        this.grid = diamondSquare(n, roughness)
        this.buffers = initBuffers(n, n);
        this.shaders = initShaders();
        const self = this;
        this.timer = new Timer(1/24, () => {
            const head = self.grid.shift();
            self.grid.push(head);
        })
    }

    update(deltaTime) {
        this.timer.update(deltaTime);
    }

    draw() {
        const rows = this.grid.length;
        const cols = this.grid[0].length;
        gl.clearColor(0.529, 0.808, 0.922, 1.0);

        gl.clearDepth(1.0);
        gl.enable(gl.DEPTH_TEST);
        gl.depthFunc(gl.LEQUAL);

        gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

        const positions = gridToPositions(this.grid);
        gl.bindBuffer(gl.ARRAY_BUFFER, this.buffers.position);
        gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(positions), gl.DYNAMIC_DRAW);
        gl.vertexAttribPointer(this.shaders.locations.attributes.aPos, 3, gl.FLOAT, false, 0, 0);
        gl.enableVertexAttribArray(this.shaders.locations.attributes.aPos);

        const normals = positionsToNormals(positions, rows, cols);
        gl.bindBuffer(gl.ARRAY_BUFFER, this.buffers.normal);
        gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(normals), gl.DYNAMIC_DRAW);
        gl.vertexAttribPointer(this.shaders.locations.attributes.aNormal, 3, gl.FLOAT, false, 0, 0)
        gl.enableVertexAttribArray(this.shaders.locations.attributes.aNormal);

        gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.buffers.index);
        gl.useProgram(this.shaders.program);

        const perspective = mat4.create();
        mat4.perspective(
            perspective,
            Math.PI / 6,
            canvas.width / canvas.height,
            0.001,
        );
        const view = mat4.create();
        const eye = vec3.fromValues(0, 2, -7);
        const at = vec3.fromValues(0, 0, 0);
        const up = vec3.fromValues(0, 1, 0);
        mat4.lookAt(view, eye, at, up);
        const model = mat4.create();

        gl.uniform3fv(this.shaders.locations.uniforms.eye, eye);
        gl.uniformMatrix4fv(this.shaders.locations.uniforms.perspective, false, perspective);
        gl.uniformMatrix4fv(this.shaders.locations.uniforms.view, false, view);
        gl.uniformMatrix4fv(this.shaders.locations.uniforms.model, false, model);

        const numElements = (rows - 1) * (cols - 1) * 2 * 3;
        gl.drawElements(gl.TRIANGLES, numElements, gl.UNSIGNED_SHORT, 0);
    }
}

function render(state) {
    state.terrain.update(state.deltaTime);
    state.terrain.draw();
}

function main() {
    let prevStep;
    const k = 7;
    const roughness = 0.45;
    const terrain = new Terrain(k, roughness);

    function mainloop(timestamp) {
        if(prevStep === undefined) {
            prevStep = timestamp;
        }
        const deltaTime = timestamp - prevStep;
        render({
            terrain: terrain,
            deltaTime: deltaTime / 1000,
        });
        prevStep = timestamp;
        requestAnimationFrame(mainloop);
    }
    requestAnimationFrame(mainloop);
}
window.onload = main;
