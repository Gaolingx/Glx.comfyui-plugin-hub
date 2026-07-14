import { app } from "../../scripts/app.js";

app.registerExtension({
	name: "INT8.PreLoraLoader",
	async beforeRegisterNodeDef(nodeType, nodeData, app) {
		if (nodeData.name === "INT8PreLoraLoader") {
			const parseSavedLoras = (values) => {
				const pairs = [];
				if (!Array.isArray(values)) {
					return pairs;
				}

				for (let i = 0; i < values.length; i++) {
					const name = values[i];
					const strength = values[i + 1];
					if (name === "Add LoRA" || name === "Remove LoRA") {
						continue;
					}
					if (typeof name === "string" && typeof strength === "number") {
						pairs.push({ name, strength });
						i++;
					}
				}
				return pairs;
			};

			const getLoraOptions = (node) => {
				for (let i = 0; i < node.widgets.length; i++) {
					const w = node.widgets[i];
					if (w && w.name && w.name.startsWith("lora_name_")) {
						return w.options?.values || [];
					}
				}
				return [];
			};

			const getStrengthOptions = (node) => {
				for (let i = 0; i < node.widgets.length; i++) {
					const w = node.widgets[i];
					if (w && w.name === "lora_strength_1") {
						const options = Object.assign({}, w.options);
						options.precision = 2;
						return {
							options,
							callback: w.callback || (() => {}),
						};
					}
				}
				return {
					options: { min: -10.0, max: 10.0, step: 0.01, precision: 2 },
					callback: () => {},
				};
			};

			const countLoraRows = (node) => {
				let maxIndex = 0;
				for (let i = 0; i < node.widgets.length; i++) {
					const match = node.widgets[i]?.name?.match(/^lora_name_(\d+)$/);
					if (match) {
						maxIndex = Math.max(maxIndex, parseInt(match[1]));
					}
				}
				return maxIndex;
			};

			const addLoraRow = (node, index, name = null, strength = 1.0) => {
				const loraOptions = getLoraOptions(node);
				const strengthConfig = getStrengthOptions(node);
				const combo = node.addWidget("combo", `lora_name_${index}`, name ?? loraOptions[0] ?? "None", () => {}, { values: loraOptions });
				const number = node.addWidget("number", `lora_strength_${index}`, strength, strengthConfig.callback, strengthConfig.options);
				return { combo, number };
			};

			const ensureLoraRows = (node, count) => {
				for (let i = countLoraRows(node) + 1; i <= count; i++) {
					addLoraRow(node, i);
				}
			};

			const applySavedLoras = (node, pairs) => {
				for (let i = 0; i < pairs.length; i++) {
					const index = i + 1;
					const nameWidget = node.widgets.find((w) => w?.name === `lora_name_${index}`);
					const strengthWidget = node.widgets.find((w) => w?.name === `lora_strength_${index}`);
					if (nameWidget) {
						nameWidget.value = pairs[i].name;
					}
					if (strengthWidget) {
						strengthWidget.value = pairs[i].strength;
					}
				}
			};

			const onNodeCreated = nodeType.prototype.onNodeCreated;
			nodeType.prototype.onNodeCreated = function () {
				const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
				
				this.updateRemoveBtn = () => {
					let maxIndex = 0;
					for (let i = 0; i < this.widgets.length; i++) {
						const w = this.widgets[i];
						if (w && w.name) {
							const match = w.name.match(/lora_name_(\d+)/);
							if (match) {
								maxIndex = Math.max(maxIndex, parseInt(match[1]));
							}
						}
					}
					
					if (maxIndex > 1) {
						if (!this.removeBtn) {
							this.removeBtn = this.addWidget("button", "Remove LoRA", "Remove LoRA", () => {
								let mIdx = 0;
								let maxNameWidget = null;
								let maxStrengthWidget = null;
								for (let i = 0; i < this.widgets.length; i++) {
									const w = this.widgets[i];
									if (w && w.name) {
										const match = w.name.match(/lora_name_(\d+)/);
										if (match) {
											const idx = parseInt(match[1]);
											if (idx > mIdx) {
												mIdx = idx;
												maxNameWidget = w;
											}
										}
										const matchStr = w.name.match(/lora_strength_(\d+)/);
										if (matchStr) {
											const idx = parseInt(matchStr[1]);
											if (idx === mIdx) {
												maxStrengthWidget = w;
											}
										}
									}
								}
								
								if (mIdx > 1) { // Never remove the first lora
									if (maxNameWidget) {
										this.widgets.splice(this.widgets.indexOf(maxNameWidget), 1);
									}
									if (maxStrengthWidget) {
										this.widgets.splice(this.widgets.indexOf(maxStrengthWidget), 1);
									}
									
									this.updateRemoveBtn();
									
									const sz = this.computeSize();
									this.size[0] = Math.max(this.size[0], sz[0]);
									this.size[1] = sz[1];
									this.setDirtyCanvas(true, true);
								}
							});
							this.removeBtn.serialize = false;
						} else {
							// Ensure it's at the bottom
							const idx = this.widgets.indexOf(this.removeBtn);
							if (idx !== -1) {
								this.widgets.splice(idx, 1);
								this.widgets.push(this.removeBtn);
							}
						}
					} else {
						if (this.removeBtn) {
							const idx = this.widgets.indexOf(this.removeBtn);
							if (idx !== -1) {
								this.widgets.splice(idx, 1);
							}
							this.removeBtn = null;
						}
					}
				};
				
				const addBtn = this.addWidget("button", "Add LoRA", "Add LoRA", () => {
					const nextIndex = countLoraRows(this) + 1;
					addLoraRow(this, nextIndex);
					
					this.updateRemoveBtn();
					
					const sz = this.computeSize();
					this.size[0] = Math.max(this.size[0], sz[0]);
					this.size[1] = Math.max(this.size[1], sz[1]);
					this.setDirtyCanvas(true, true);
				});
				addBtn.serialize = false;
				this.addBtn = addBtn;
				
				// Move addBtn to top
				this.widgets.splice(this.widgets.indexOf(addBtn), 1);
				this.widgets.unshift(addBtn);

				// Fix the precision of the initial lora_strength_1 to display 2 decimals
				for (let i = 0; i < this.widgets.length; i++) {
					if (this.widgets[i] && this.widgets[i].name === "lora_strength_1") {
						this.widgets[i].options.precision = 2;
					}
				}
				
				// Initially call update RemoveBtn to ensure correct state (should be hidden)
				this.updateRemoveBtn();
				
				return r;
			};
			
			const onConfigure = nodeType.prototype.onConfigure;
			nodeType.prototype.onConfigure = function (info) {
				const savedLoras = parseSavedLoras(info?.widgets_values);
				if (info && info.widgets_values) {
					ensureLoraRows(this, savedLoras.length);
					if (this.updateRemoveBtn) {
						this.updateRemoveBtn();
					}
				}
				
				const r = onConfigure ? onConfigure.apply(this, arguments) : undefined;
				applySavedLoras(this, savedLoras);
				if (this.addBtn) {
					this.addBtn.value = "Add LoRA";
				}
				if (this.removeBtn) {
					this.removeBtn.value = "Remove LoRA";
				}
				
				if (this.updateRemoveBtn) {
					this.updateRemoveBtn();
				}
				const sz = this.computeSize();
				this.size[0] = Math.max(this.size[0], sz[0]);
				this.size[1] = sz[1];
				return r;
			};
		}
	}
});
