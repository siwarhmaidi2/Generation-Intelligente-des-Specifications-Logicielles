import argparse
import json
from pathlib import Path
from agent_architect import AgentArchitect

def iter_json_files(input_dir: Path) -> list[Path]:
    return sorted([p for p in input_dir.glob("*.json") if not p.name.endswith("_architecture.json")])

def main() -> None:
    parser = argparse.ArgumentParser(description="Requirements JSON -> Architecture JSON")
    parser.add_argument("--input-dir", type=str, default="results")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--file", type=str, help="Traiter un fichier spécifique uniquement")
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    
    json_files = iter_json_files(input_dir)
    
    if args.file:
        json_files = [p for p in json_files if p.name == args.file]
        
    if not json_files:
        print(f"❌ Aucun fichier JSON de requirements trouvé dans {input_dir}")
        return

    agent = AgentArchitect()
    
    for json_file in json_files:
        out_file = input_dir / f"{json_file.stem}_architecture.json"
        
        if out_file.exists() and not args.overwrite:
            print(f"⏭️ Skip : {json_file.name} (existe déjà)")
            continue
            
        print(f"🏗️ Génération Architecture pour : {json_file.name}")
        
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                requirements = json.load(f)
        except Exception as e:
            print(f"⚠️ Erreur lecture JSON {json_file.name}: {e}")
            continue

        arch_analysis = agent.generate_architecture(requirements)
        
        with open(out_file, "w", encoding="utf-8") as f:
            f.write(arch_analysis.model_dump_json(indent=2, ensure_ascii=False, exclude={"raw_model_output"}))
            
        print(f"✅ OK : {out_file}")
        
        # Optionnel: Sauvegarder les diagrammes UML dans des fichiers séparés
        uml_dir = input_dir / "uml" / json_file.stem
        uml_dir.mkdir(parents=True, exist_ok=True)
        
        for idx, uml in enumerate(arch_analysis.uml_diagrams):
            uml_path = uml_dir / f"{uml.diagram_type}_{idx}.puml"
            with open(uml_path, "w", encoding="utf-8") as f:
                f.write(uml.content)
            print(f"   📄 Diagramme sauvegardé : {uml_path}")

    print("\n🎉 Architecture terminée")

if __name__ == "__main__":
    main()
