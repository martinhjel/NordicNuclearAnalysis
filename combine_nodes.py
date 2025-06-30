from pathlib import Path

import pandas as pd


class PowerGamaDataLoader:
    """
    Loads PowerGama input CSV files for a given dataset configuration.

    :param year: Dataset year (e.g., 2025)
    :param scenario: Scenario name (e.g., "BM")
    :param version: Dataset version string (e.g., "100")
    :param base_path: Base path to the dataset folder
    """

    def __init__(self, year: int, scenario: str, version: str, base_path: Path) -> None:
        self.dataset_path = base_path / f"CASE_{year}/scenario_{scenario}/data/system"
        self.version = version
        self.scenario = scenario

        self.files = {
            "branch": self.dataset_path / f"branch_{scenario}_v{version}.csv",
            "generator": self.dataset_path / f"generator_{scenario}_v{version}.csv",
            "consumer": self.dataset_path / f"consumer_{scenario}_v{version}.csv",
            "dcbranch": self.dataset_path / f"dcbranch_{scenario}_v{version}.csv",
            "node": self.dataset_path / f"node_{scenario}_v{version}.csv",
        }

        for name, file in self.files.items():
            if not file.exists():
                raise FileNotFoundError(f"{name} file not found: {file}")

        self._dfs = {name: pd.read_csv(path, index_col=0) for name, path in self.files.items()}

    @property
    def branch(self) -> pd.DataFrame:
        return self._dfs["branch"]

    @property
    def generator(self) -> pd.DataFrame:
        return self._dfs["generator"]

    @property
    def consumer(self) -> pd.DataFrame:
        return self._dfs["consumer"]

    @property
    def dcbranch(self) -> pd.DataFrame:
        return self._dfs["dcbranch"]

    @property
    def node(self) -> pd.DataFrame:
        return self._dfs["node"]


def simplify_node(node: str) -> str:
    """Strip trailing underscore and digits (e.g., DK1_3 -> DK1)."""
    return node.split("_")[0]


def combine_external_branch(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Simplify node names
    df["node_from_simple"] = df["node_from"].apply(simplify_node)
    df["node_to_simple"] = df["node_to"].apply(simplify_node)

    # Remove internal connections (e.g., DK1_1 to DK1_2)
    df = df[df["node_from_simple"] != df["node_to_simple"]]

    # Group external connections
    grouped = df.groupby(["node_from_simple", "node_to_simple"])

    # Aggregate: sum capacity, average impedances
    combined = grouped.agg(
        {
            "capacity": "sum",
            "reactance_ohm": "mean",
            "resistance_ohm": "mean",
            "reactance": "mean",
            "resistance": "mean",
        }
    ).reset_index()

    combined.rename(columns={"node_from_simple": "node_from", "node_to_simple": "node_to"}, inplace=True)

    return combined


def process_branches(branch_df: pd.DataFrame) -> pd.DataFrame:
    branch_df = branch_df.copy()

    # Simplify node names for branch endpoints
    branch_df["node_from"] = branch_df["node_from"].apply(simplify_node)
    branch_df["node_to"] = branch_df["node_to"].apply(simplify_node)

    # Remove internal subnode-to-subnode connections (same simplified node)
    branch_df = branch_df[branch_df["node_from"] != branch_df["node_to"]]

    # Group by direction and aggregate capacity
    grouped = branch_df.groupby(["node_from", "node_to"]).agg({"capacity": "sum", "resistance": "mean"}).reset_index()

    # Add placeholder values for reactance to match AC line columns
    grouped["reactance"] = 0.0
    grouped["reactance_ohm"] = 0.0
    grouped["resistance_ohm"] = grouped["resistance"]

    return grouped


def combine_generators(df):
    # Extract main node (e.g., DK1 from DK1_1)
    df["main_node"] = df["node"].str.extract(r"^([^_]+)")

    # Define column categories
    sum_cols = ["pmax", "pmin", "storage_cap", "pump_cap"]
    av_cols = [
        "fuelcost",
        "inflow_fac",
        "storage_ini",
        "storage_price",
        "pump_efficiency",
        "pump_deadband",
        "gen_lat",
        "gen_lon",
    ]
    fixed_cols = ["desc", "inflow_ref", "type", "year", "status", "source", "storval_filling_ref", "storval_time_ref"]

    # Build aggregation rules based on available columns
    agg_rules = {}
    for col in sum_cols:
        if col in df.columns:
            agg_rules[col] = "sum"
    for col in av_cols:
        if col in df.columns:
            agg_rules[col] = "mean"
    for col in fixed_cols:
        if col in df.columns:
            agg_rules[col] = "first"

    # Perform aggregation
    agg_df = df.groupby(["main_node", "type"], as_index=False).agg(agg_rules)

    # Rename back to "node"
    agg_df.rename(columns={"main_node": "node"}, inplace=True)

    # Regenerate consistent description
    agg_df["desc"] = agg_df["node"] + " " + agg_df["type"]

    return agg_df


def combine_consumers(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Drop unnamed columns
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]

    # Simplify node names
    df["node_simple"] = df["node"].apply(simplify_node)

    # List of columns to sum or average
    sum_cols = ["demand_avg"]
    mean_cols = [
        "flex_fraction",
        "flex_on_off",
        "flex_basevalue",
        "flex_storage",
        "flex_storval_filling",
        "flex_storval_time",
        "flex_storagelevel_init",
    ]

    # Group and aggregate
    agg_dict = {col: "sum" for col in sum_cols}
    agg_dict.update({col: "mean" for col in mean_cols})
    agg_dict.update({"demand_ref": "first", "status": "first"})

    grouped = df.groupby("node_simple").agg(agg_dict).reset_index()
    grouped.rename(columns={"node_simple": "node"}, inplace=True)

    # Add Load column = node name
    grouped["Load"] = grouped["node"]

    # Reorder columns
    cols = ["Load", "node"] + [col for col in grouped.columns if col not in ["Load", "node"]]
    grouped = grouped[cols]

    return grouped


def combine_nodes(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Simplify node names
    df["id"] = df["id"].apply(simplify_node)

    # Group external connections
    grouped = df.groupby(["id"])

    # Aggregate: sum capacity, average impedances
    combined = grouped.agg(
        {
            "lat": "mean",
            "lon": "mean",
            "area": "first",
            "zone": "first",
        }
    ).reset_index()

    return combined


if __name__ == "__main__":
    dataset_year = 2025
    dataset_scenario = "BM"
    dataset_version = "100"
    base_path = Path().cwd()
    dataset_path = base_path / f"CASE_{dataset_year}/scenario_{dataset_scenario}/data/system"

    output_path_node = dataset_path / f"combined/node_{dataset_scenario}_v{dataset_version}.csv"
    output_path_branch = dataset_path / f"combined/branch_{dataset_scenario}_v{dataset_version}.csv"
    output_path_consumers = dataset_path / f"combined/consumer_{dataset_scenario}_v{dataset_version}.csv"
    output_path_generators = dataset_path / f"combined/generator_{dataset_scenario}_v{dataset_version}.csv"

    output_path_generators.parent.mkdir(exist_ok=True)

    data_loader = PowerGamaDataLoader(
        year=dataset_year, scenario=dataset_scenario, version=dataset_version, base_path=base_path
    )

    combined_nodes = combine_nodes(data_loader.node)
    combined_nodes.to_csv(output_path_node)

    combined_ac = combine_external_branch(data_loader.branch)
    combined_branches = process_branches(data_loader.dcbranch)
    full_network = pd.concat([combined_ac, combined_branches], ignore_index=True)
    full_network.sort_values(["node_from"])
    full_network.to_csv(output_path_branch)

    combined_consumer = combine_consumers(data_loader.consumer)
    combined_consumer.to_csv(output_path_consumers)

    combined_generators = combine_generators(data_loader.generator)
    combined_generators.to_csv(output_path_generators)
