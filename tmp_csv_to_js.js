const fs = require('fs');

const csvPath = 'python-engine/data/puzzle_ml_dataset_700.csv';
const outPath = 'src/data.js';

const csvData = fs.readFileSync(csvPath, 'utf8');
const lines = csvData.split('\n').filter(l => l.trim().length > 0);

const headers = lines[0].split(',');
const dataRows = [];

for (let i = 1; i < lines.length; i++) {
    // Basic regex split that handles quotes, though our data seems clean except for puzzle_state
    // Actually, puzzle_state has quotes and commas like "(2, 0, 3, 1, 4, 6, 7, 5, 8)"
    const rowStr = lines[i];
    
    // We only need specific features and the label.
    // manhattan_distance,misplaced_tiles,linear_conflict,corner_misplaced,
    // blank_position,blank_row,blank_col,blank_in_center,
    // max_tile_displacement,num_valid_moves ... difficulty_label
    
    // Let's parse properly.
    const parts = rowStr.split(/,(?=(?:(?:[^"]*"){2})*[^"]*$)/);
    
    if(parts.length < headers.length) continue;
    
    const obj = {};
    for (let j = 0; j < headers.length; j++) {
        let val = parts[j].replace(/^"|"$/g, '').trim();
        // Convert all features to floats except puzzle_id, puzzle_state, difficulty_label
        if (headers[j] !== 'puzzle_state' && headers[j] !== 'difficulty_label' && headers[j] !== 'puzzle_id') {
            val = parseFloat(val);
        }
        obj[headers[j]] = val;
    }
    
    // As per user prompt, features = 10 specific headers:
    const features = {
        manhattan_distance: obj.manhattan_distance,
        misplaced_tiles: obj.misplaced_tiles,
        linear_conflict: obj.linear_conflict,
        corner_misplaced: obj.corner_misplaced,
        blank_position: obj.blank_position,
        blank_row: obj.blank_row,
        blank_col: obj.blank_col,
        blank_in_center: obj.blank_in_center,
        max_tile_displacement: obj.max_tile_displacement,
        num_valid_moves: obj.num_valid_moves
    };
    
    dataRows.push({
        features: features,
        label: obj.difficulty_label
    });
}

const jsContent = `// Auto-generated 700-row puzzle dataset
const TRAINING_DATA = ${JSON.stringify(dataRows, null, 2)};
`;

fs.writeFileSync(outPath, jsContent, 'utf8');
console.log('Successfully wrote src/data.js with ' + dataRows.length + ' records.');
