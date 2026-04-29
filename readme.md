# KoboldAI-based Agent
## Project Outline
This project is based around designing an endpoint for the native KoboldCPP API (http://localhost:5001/api/) that is able to complete a task outlined in a file named `task.md` on its' own (with human supervision).

## Permissions

1. Read/write context memory
 - The model can read/write/delete up to the top 1/8 of the context (2048 of 16384 tokens (n_ctx) in this instance) on a per-line basis.
 - Perhaps the model could use a command (i.e. `/cmem w` or `/cmem d`) where it would then be prompted to enter the line it wants to modify, delete, or overwrite, and then be prompted what to over/write.

2. Read/write persistent memory
 - The model can read/write/delete an indefinite amount of data as it needs to a special `memory.md` file on a per-line basis, like context memory but saved to a file.
 - Because context memory uses the top 2048 tokens (n_ctx), that leaves the model with 14336 tokens for reading at a time. Commands could be `/pgup` and `/pgdown` for scrolling.

3. Read/write files in working directory and subdirectories
 - The model can read/write/delete files.
 - Works similarly to R/W pers. mem., but command syntax is different (`/dir {location}`, `/write {file}`, `/read {file}`. `/del {file}`; etc.).

4. Search and browse the web
 - The model can search the web and browse webpages to gather info.
 - This traffic MUST pass through a socks5 proxy present at localhost:9050.
 - Perhaps commands like `/search "{query}"` and `/goto {URL}` would be in order.

## Important KCPP API tools

 1. [POST] /api/extra/generate/stream
 - Generates text given a prompt and generation settings, with SSE streaming.
 - Unspecified values are set to defaults.

Request body example:
```json
{"max_context_length":16384,
	"max_length":128,
	"rep_pen":1.05,
	"temperature":0.5,
	"top_p":0.9,
	"rep_pen_range":360,
	"rep_pen_slope":0.7,
	"genkey":KCPP0001
	"sampler_order":[6,0,1,3,4,2,5],
	"memory":"Memory goes here!\n",
	"stop_sequence":[],
	"prompt": "Niko the kobold stalked carefully down the alley, his small scaly figure obscured by a dusky cloak that fluttered lightly in the cold winter breeze."}
```

Request return example:
```json
event: message
data: {"token": " His", "finish_reason": null}

event: message
data: {"token": " eyes", "finish_reason": null}

event: message
data: {"token": " gle", "finish_reason": null}

event: message
data: {"token": "amed", "finish_reason": null}

event: message
data: {"token": " with", "finish_reason": null}

event: message
data: {"token": " a", "finish_reason": null}

event: message
data: {"token": " sharp", "finish_reason": null}

event: message
data: {"token": ",", "finish_reason": null}

...

event: message
data: {"token": " disappeared", "finish_reason": null}

event: message
data: {"token": " into", "finish_reason": null}

event: message
data: {"token": " the", "finish_reason": null}

event: message
data: {"token": " night", "finish_reason": null}

event: message
data: {"token": ".", "finish_reason": null}

event: message
data: {"token": "", "finish_reason": "stop"}
```

2. [POST] /api/extra/abort
 - Aborts a generation.

Request body example:
```json
{"genkey":"KCPP0001"}
```

Request response example:
```json
{"success": "true", "done": "true"}
```

3. [POST] /api/extra/tokenize
 - Counts the number of tokens in a string, and returns their token IDs.

Request body example:
```json
{"prompt": "Hello, my name is Niko."}
```

Request response example:
```json
{"value": 9,
	"ids": [
		1,
		22557,
		28725,
		586,
		1141,
		349,
		11952,
		28709,
		28723
	]
}
```

4. [POST] /api/extra/detokenize
 - Converts an array of token IDs into a string.

Request body example:
```json
{
	"ids": [
		529,
		29988,
		5205,
		29989,
		29958,
		13
	]
}
```

Request response example:
```json
{
	"result": "’ TamPORT Packet Interior.",
	"success": true
}
```

## One More Thing
Feel free to update and modify this file to be accurate to current/future versions.
