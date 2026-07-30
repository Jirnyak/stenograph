const https = require('https');
const fs = require('fs');

const token = 'ghp_REDACTED_TOKEN_PLACEHOLDER';
const repo = 'Jirnyak/stenograph';
const path = 'index.html';

const getOptions = {
    hostname: 'api.github.com',
    path: `/repos/${repo}/contents/${path}`,
    method: 'GET',
    headers: {
        'User-Agent': 'Node.js',
        'Authorization': `token ${token}`,
        'Accept': 'application/vnd.github.v3+json'
    }
};

let sha = null;

const getReq = https.request(getOptions, (res) => {
    let data = '';
    res.on('data', chunk => data += chunk);
    res.on('end', () => {
        if (res.statusCode === 200) {
            const json = JSON.parse(data);
            sha = json.sha;
        }
        pushFile();
    });
});

getReq.on('error', (e) => {
    console.error('Error fetching existing file:', e);
    pushFile();
});
getReq.end();

function pushFile() {
    const content = fs.readFileSync('index.html', 'utf8');
    const contentBase64 = Buffer.from(content).toString('base64');
    
    const body = JSON.stringify({
        message: 'Redesign index.html with premium retro-modern typewriter SPA',
        content: contentBase64,
        sha: sha
    });

    const putOptions = {
        hostname: 'api.github.com',
        path: `/repos/${repo}/contents/${path}`,
        method: 'PUT',
        headers: {
            'User-Agent': 'Node.js',
            'Authorization': `token ${token}`,
            'Accept': 'application/vnd.github.v3+json',
            'Content-Type': 'application/json',
            'Content-Length': Buffer.byteLength(body)
        }
    };

    const putReq = https.request(putOptions, (res) => {
        let data = '';
        res.on('data', chunk => data += chunk);
        res.on('end', () => {
            console.log('Push Status:', res.statusCode);
            if (res.statusCode === 200 || res.statusCode === 201) {
                console.log('Successfully pushed index.html to GitHub!');
                const json = JSON.parse(data);
                if(json.content && json.content.html_url) {
                    console.log('URL:', json.content.html_url);
                }
            } else {
                console.error('Failed to push file. Response:', data);
            }
        });
    });

    putReq.on('error', (e) => {
        console.error('Error pushing file:', e);
    });
    putReq.write(body);
    putReq.end();
}
