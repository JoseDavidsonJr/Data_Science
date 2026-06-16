import asyncio
import json
import sys
from pathlib import Path
from mcp_brasil.data.tce_pe.client import buscar_despesas

UNIT_NAME = "Autarquia Educacional do Vale do São Francisco de Petrolina"
MUNICIPIO_CODE = "115"

async def fetch_year(ano):
    print(f"Buscando dados de {ano}...")
    all_despesas = []
    # TCE-PE Sagres usually requires month-by-month or returns a lot.
    # The client allows mes=None, but let's do it month by month for safety if needed,
    # or just try mes=None first.
    try:
        # Try fetching the whole year if possible
        despesas = await buscar_despesas(ano=ano, codigo_municipio=MUNICIPIO_CODE)
        all_despesas = [d for d in despesas if d.unidade_gestora == UNIT_NAME]
    except Exception as e:
        print(f"Erro ao buscar ano {ano}: {e}")
        # Fallback to month by month
        for mes in range(1, 13):
            try:
                print(f"  Buscando mês {mes}...")
                despesas = await buscar_despesas(ano=ano, mes=mes, codigo_municipio=MUNICIPIO_CODE)
                filtered = [d for d in despesas if d.unidade_gestora == UNIT_NAME]
                all_despesas.extend(filtered)
            except Exception as e2:
                print(f"    Erro no mês {mes}: {e2}")
    
    return all_despesas

def map_to_original_format(despesas):
    mapped = []
    for d in despesas:
        mapped.append({
            "id": d.numero_empenho,
            "fornecedor": {
                "pessoa": {
                    "nome": d.fornecedor,
                    "cpfCnpj": d.cpf_cnpj
                },
                "nome": d.fornecedor
            },
            "valorEmpenhado": d.valor_empenhado,
            "valorLiquidado": d.valor_liquidado,
            "valorPago": d.valor_pago,
            "valorAnulado": 0.0, # Not directly in Despesa object?
            "valorRetido": 0.0,
            "valorSaldoAPagar": (d.valor_empenhado or 0) - (d.valor_pago or 0),
            "historico": d.historico
        })
    return mapped

async def main(anos):
    for ano in anos:
        despesas = await fetch_year(ano)
        print(f"Encontradas {len(despesas)} despesas para {ano}.")
        if despesas:
            mapped = map_to_original_format(despesas)
            output_file = Path(f"despesas{str(ano)[2:]}_mcp.JSON")
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(mapped, f, ensure_ascii=False, indent=2)
            print(f"Salvo em {output_file}")

if __name__ == "__main__":
    anos_to_fetch = [int(a) for a in sys.argv[1:]] if len(sys.argv) > 1 else [2024]
    asyncio.run(main(anos_to_fetch))
