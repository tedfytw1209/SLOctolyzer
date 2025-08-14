import argparse
from sloctolyzer import main_dicom

def main():
    parser = argparse.ArgumentParser(description="Run main_dicom with custom parameters.")

    parser.add_argument("--analysis_csv", default=r"/blue/ruogu.fang/tienyuchang/OCT_EDA/Paired_OCT_Data_ADCON_samples_part2.csv", help="Path to the analysis CSV file")
    parser.add_argument("--output_directory", default=r"/blue/ruogu.fang/tienyuchang/SLO_Output2", help="Directory to save the output")
    parser.add_argument("--robust_run", type=int, default=1)
    parser.add_argument("--save_individual_segmentations", type=int, default=0)

    args = parser.parse_args()

    main_dicom.run(vars(args))

if __name__ == "__main__":
    main()