import math


def calculate_p_chart_limits(
    defects: list[int],
    sample_sizes: list[int]
) -> tuple[float, list[float], list[float]]:
    """
    Calculates the Control Limits for a p-chart (proportion of defective units)
    with variable sample sizes.

    Args:
        defects: A list containing the number of defective units in each sample.
        sample_sizes: A list containing the total number of units in each sample.

    Returns:
        A tuple containing:
            - center_line (p_bar): Overall proportion of defects.
            - ucl: A list of Upper Control Limits for each sample.
            - lcl: A list of Lower Control Limits for each sample.
    """
    if len(defects) != len(sample_sizes):
        raise ValueError("The length of defects and sample_sizes must be equal.")
    if sum(sample_sizes) == 0:
        return 0.0, [], []

    total_defects = sum(defects)
    total_samples = sum(sample_sizes)

    # Center Line (p_bar)
    p_bar = total_defects / total_samples

    ucl = []
    lcl = []

    for n_i in sample_sizes:
        if n_i == 0:
            ucl.append(0.0)
            lcl.append(0.0)
            continue

        std_dev = math.sqrt((p_bar * (1 - p_bar)) / n_i)

        # UCL = p_bar + 3 * std_dev
        current_ucl = p_bar + 3 * std_dev
        # LCL = p_bar - 3 * std_dev, bounded by 0
        current_lcl = max(0.0, p_bar - 3 * std_dev)

        ucl.append(current_ucl)
        lcl.append(current_lcl)

    return p_bar, ucl, lcl


def calculate_u_chart_limits(
    defects: list[int],
    units: list[float]
) -> tuple[float, list[float], list[float]]:
    """
    Calculates the Control Limits for a u-chart (defects per unit)
    with variable unit sizes (e.g. lines of code / 100).

    Args:
        defects: A list containing the number of defects found.
        units: A list containing the 'size' or 'area of opportunity' for each sample.

    Returns:
        A tuple containing:
            - center_line (u_bar): Overall defects per unit.
            - ucl: A list of Upper Control Limits for each sample.
            - lcl: A list of Lower Control Limits for each sample.
    """
    if len(defects) != len(units):
        raise ValueError("The length of defects and units must be equal.")
    if sum(units) == 0:
        return 0.0, [], []

    total_defects = sum(defects)
    total_units = sum(units)

    # Center Line (u_bar)
    u_bar = total_defects / total_units

    ucl = []
    lcl = []

    for n_i in units:
        if n_i == 0:
            ucl.append(0.0)
            lcl.append(0.0)
            continue

        std_dev = math.sqrt(u_bar / n_i)

        # UCL = u_bar + 3 * std_dev
        current_ucl = u_bar + 3 * std_dev
        # LCL = u_bar - 3 * std_dev, bounded by 0
        current_lcl = max(0.0, u_bar - 3 * std_dev)

        ucl.append(current_ucl)
        lcl.append(current_lcl)

    return u_bar, ucl, lcl
