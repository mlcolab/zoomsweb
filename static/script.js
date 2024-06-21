const fileInput = document.getElementById('fileInput');
const fileTree = document.getElementById('fileTree');
const fileContent = document.getElementById('fileContent');
const plot = document.getElementById('plot');
const resizer = document.getElementById('resizer');
const fileListContainer = document.getElementById('fileListContainer');
const plotContainer = document.getElementById('plotContainer');
const resultList = document.getElementById('resultList');

let currentPath = [];
let fileSystem = {};
let lastCSVData = null; // To store the last loaded CSV data

fileInput.addEventListener('change', (event) => {
    const files = Array.from(event.target.files);
    fileSystem = buildFileTree(files);
    currentPath = [];
    renderFileTree(fileSystem, currentPath);
});

function buildFileTree(files) {
    const root = {};
    files.forEach(file => {
        const parts = file.webkitRelativePath.split('/');
        let current = root;
        parts.forEach((part, index) => {
            if (!current[part]) {
                current[part] = index === parts.length - 1 ? file : {};
            }
            current = current[part];
        });
    });
    return root;
}

function renderFileTree(tree, path) {
    fileTree.innerHTML = '';

    let currentFolder = tree;
    path.forEach(folder => {
        currentFolder = currentFolder[folder];
    });

    if (path.length > 0) {
        const backItem = document.createElement('div');
        backItem.textContent = '..';
        backItem.classList.add('back', 'cursor-pointer', 'my-1', 'p-1', 'rounded');
        backItem.addEventListener('click', () => {
            path.pop();
            renderFileTree(tree, path);
            plot.innerHTML = '';  // Clear the plot
            fileContent.classList.add('hidden');  // Hide the file content
        });
        fileTree.appendChild(backItem);
    }

    Object.keys(currentFolder).forEach(key => {
        const item = document.createElement('div');
        if (currentFolder[key] instanceof File) {
            item.textContent = key;
            item.classList.add('file', 'cursor-pointer', 'my-1', 'p-1', 'rounded');
            item.addEventListener('click', () => {
                const file = currentFolder[key];
                if (file.name.endsWith('.csv')) {
                    const reader = new FileReader();
                    reader.onload = async (e) => {
                        lastCSVData = e.target.result;
                        plotRawCSV(lastCSVData);
                        const preprocessedData = preprocessCSV(lastCSVData);
                        await runModel(preprocessedData);
                    };
                    reader.readAsText(file);
                } else {
                    const reader = new FileReader();
                    reader.onload = (e) => {
                        fileContent.textContent = e.target.result;
                        fileContent.classList.remove('hidden');
                        plot.innerHTML = '';  // Clear the plot
                    };
                    reader.readAsText(file);
                }
            });
        } else {
            item.textContent = key;
            item.classList.add('folder', 'cursor-pointer', 'my-1', 'p-1', 'rounded');
            item.addEventListener('click', () => {
                path.push(key);
                renderFileTree(tree, path);
                plot.innerHTML = '';  // Clear the plot
                fileContent.classList.add('hidden');  // Hide the file content
            });
        }
        fileTree.appendChild(item);
    });
}

function preprocess(intensities, masses, binResolution = 0.5) {
    const bins = [];
    for (let i = 899.9; i < 3500; i += binResolution) {
        bins.push(i);
    }
    const binCount = bins.length;
    const binSums = new Array(binCount).fill(0);
    const binCounts = new Array(binCount).fill(0);

    const binIndices = masses.map(mass => bins.findIndex(bin => mass < bin));

    for (let i = 0; i < binIndices.length; i++) {
        const binIndex = binIndices[i] - 1; // Adjusting for 0-based index
        if (binIndex >= 0 && binIndex < binCount) {
            binSums[binIndex] += intensities[i];
            binCounts[binIndex] += 1;
        }
    }

    const binMeans = binSums.map((sum, index) => binCounts[index] !== 0 ? sum / binCounts[index] : 0);

    // Normalize the results
    const mean = binMeans.reduce((acc, val) => acc + val, 0) / binMeans.length;
    const std = Math.sqrt(binMeans.reduce((acc, val) => acc + Math.pow(val - mean, 2), 0) / binMeans.length);
    const normalizedMeans = binMeans.map(val => (val - mean) / (std + Number.EPSILON));

    // Calculate the preprocessed mass axis (midpoints of bins)
    const binMidpoints = bins.map((bin, index) => index < binCount - 1 ? (bins[index] + bins[index + 1]) / 2 : null).slice(0, -1);

    return { normalizedMeans, binMidpoints };
}

function plotRawCSV(data) {
    const rows = data.split('\n').slice(1).map(row => row.split(',').map(Number));
    const mass = rows.map(row => row[0]);
    const intensity = rows.map(row => row[1]);

    const trace = {
        x: mass,
        y: intensity,
        mode: 'lines',
        type: 'scatter'
    };

    const layout = {
        title: 'Mass vs Intensity (Raw Data)',
        xaxis: { title: 'Mass' },
        yaxis: { title: 'Intensity' }
    };

    Plotly.newPlot('plot', [trace], layout);
}

function preprocessCSV(data) {
    const rows = data.split('\n').slice(1).map(row => row.split(',').map(Number));
    const mass = rows.map(row => row[0]);
    const intensity = rows.map(row => row[1]);

    const result = preprocess(intensity, mass);
    return result.normalizedMeans;
}

async function runModel(data) {
    const response = await fetch('/run_model', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ data })
    });
    const results = await response.json();
    displayResults(results);
}

function displayResults(results) {
    resultList.innerHTML = '<h3>Classification Results:</h3>';
    results.forEach(result => {
        const item = document.createElement('div');
        item.textContent = `${result.name}: ${result.score.toFixed(4)}`;
        resultList.appendChild(item);
        console.log(`${result.name}: ${result.peaks}`);
    });
}

// Draggable resizer
resizer.addEventListener('mousedown', function(e) {
    e.preventDefault();
    document.addEventListener('mousemove', resize);
    document.addEventListener('mouseup', stopResize);
});

function resize(e) {
    const fileListWidth = e.clientX;
    const plotWidth = window.innerWidth - fileListWidth - resizer.offsetWidth;

    if (fileListWidth > 50 && plotWidth > 50) { // Ensure minimum widths for both containers
        fileListContainer.style.width = fileListWidth + 'px';
        plotContainer.style.width = plotWidth + 'px';
    }
}

function stopResize() {
    document.removeEventListener('mousemove', resize);
    document.removeEventListener('mouseup', stopResize);
}
