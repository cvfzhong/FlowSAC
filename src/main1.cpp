
#include"flowsac.h"
#include"utils.h"
#include<time.h>
#include<string>

int main(int argc, char *argv[])
{
	if (argc < 8)
	{
		printf("Example: flowsac1 src.png tar.png match.txt ucm.png sed.edge outputFlow.flo sintel -L 3 -ucmT 0.1 -ransacT 2.6 -score SOD -fast -r 10\n");
		return 0;
	}
	std::string srcFile = argv[1], tarFile = argv[2], matchFile=argv[3], ucmFile = argv[4], edgeFile = argv[5], outFile=argv[6], datasetName = argv[7];
	
	try
	{
		ff::ArgSet args;
		args.setArgs(argc - 8, argv + 8, false);

		time_t beg = clock();
		CFlowSAC flowSAC;
		Mat2f flow = flowSAC(srcFile, tarFile, matchFile, ucmFile, edgeFile, datasetName, args);
		printf("time=%.2lf\n", double(clock()-beg)/CLOCKS_PER_SEC);

		writeFlowFile(flow, outFile);

		//compute and show AEE if the groundtruth is provided
		std::string gtFile = args.getd<std::string>("gt", "");
		if (!gtFile.empty())
		{
			Mat1b gtMask;
			Mat2f gt;
			readFlowX(gtFile, gt, gtMask);
			double aee = getAEE(gt, flow, gtMask);
			printf("aee=%.3lf\n", aee);
		}
	}
	catch (const std::exception &ec) {
		printf("error:%s\n", ec.what());
	}
	catch(...)
	{ }
	
	return 0;
}

