import argparse
import asyncio
import json
from pathlib import Path
from agent_product_owner import AgentProductOwner

# =====================================================
# Async execution
# =====================================================

async def run_product_owner(input_filename: str):
    # Chemin complet du fichier d'entrée dans results/
    input_path = Path("results") / input_filename

    if not input_path.exists():
        print(f"❌ Fichier introuvable : {input_path}")
        return

    print(f"📥 Chargement des exigences : {input_path}")

    with open(input_path, "r", encoding="utf-8") as f:
        requirements = json.load(f)

    print("🤖 Initialisation Agent Product Owner...")
    agent = AgentProductOwner()

    # Génération du backlog Agile
    print("🚀 Génération du backlog Agile...")
    po_output = await agent.generate_product_owner(requirements)

    # Convertir en dict pour JSON
    result_dict = po_output.model_dump() if hasattr(po_output, "model_dump") else po_output

    # =================================================
    # Créer le dossier results/product_owner
    # =================================================
    results_dir = Path("results/product_owner")
    results_dir.mkdir(parents=True, exist_ok=True)

    # Nom du fichier de sortie basé sur le fichier d'entrée
    output_filename = f"{input_path.stem}_product_owner.json"
    output_path = results_dir / output_filename

    # Sauvegarder le fichier
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result_dict, f, indent=2, ensure_ascii=False)

    print("✅ Backlog généré avec succès !")
    print(f"📁 Fichier sauvegardé : {output_path}")

    # Résumé console
    print("\n📊 Résumé :")
    if hasattr(po_output, "vision"):
        print(f"Vision produit : {po_output.vision}")
    if hasattr(po_output, "epics"):
        print(f"Nombre d'Epics : {len(po_output.epics)}")
    if hasattr(po_output, "user_stories"):
        print(f"Nombre de User Stories : {len(po_output.user_stories)}")
    if hasattr(po_output, "sprints"):
        print(f"Nombre de Sprints : {len(po_output.sprints)}")


# =====================================================
# CLI Entry
# =====================================================
def main():
    parser = argparse.ArgumentParser(
        description="Generate Agile Product Backlog using Product Owner Agent"
    )

    parser.add_argument(
        "--file",
        required=True,
        help="Nom du fichier JSON dans results/ (ex: meeting.json)"
    )

    args = parser.parse_args()

    asyncio.run(run_product_owner(args.file))


if __name__ == "__main__":
    main()