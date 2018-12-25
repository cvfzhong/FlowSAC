#include "RIC.h"
#include "Image.h"
#include "OpticFlowIO.h"
#include "opencv2/ximgproc.hpp" // for struct edge detection

namespace {

	/* read semi-dense matches, stored as x1 y1 x2 y2 per line (other values on the same line is not taking into account) */
	void ReadMatches(const char *filename, FImage& outMat)
	{
		float* tmp = new float[4 * 100000]; // max number of match pair
		FILE *fid = fopen(filename, "r");
		int nmatch = 0;
		float x1, x2, y1, y2;
		while (!feof(fid) && fscanf(fid, "%f %f %f %f%*[^\n]", &x1, &y1, &x2, &y2) == 4) {
			tmp[4 * nmatch] = x1;
			tmp[4 * nmatch + 1] = y1;
			tmp[4 * nmatch + 2] = x2;
			tmp[4 * nmatch + 3] = y2;
			nmatch++;
		}
		outMat.allocate(4, nmatch);
		memcpy(outMat.pData, tmp, nmatch * 4 * sizeof(float));
		fclose(fid);
		delete[] tmp;
	}

	// prepare cost map from Structured Edge Detector
	void GetCostMap(char* imgName, FImage& outCostMap, const std::string &sedModelFile)
	{
		cv::Mat cvImg1 = cv::imread(imgName);
		int w = cvImg1.cols;
		int h = cvImg1.rows;
		outCostMap.allocate(w, h, 1);

		cv::Mat fImg1;
		// convert source image to [0-1] range
		cvImg1.convertTo(fImg1, cv::DataType<float>::type, 1 / 255.0);
		int borderSize = 10;
		cv::copyMakeBorder(fImg1, fImg1, borderSize, borderSize, borderSize, borderSize, cv::BORDER_REPLICATE);
		cv::Mat edges(fImg1.size(), fImg1.type());
		static cv::Ptr<cv::ximgproc::StructuredEdgeDetection> sEdge = cv::ximgproc::createStructuredEdgeDetection(sedModelFile.c_str());
		sEdge->detectEdges(fImg1, edges);
		// save result to FImage
		for (int i = 0; i < h; i++) {
			for (int j = 0; j < w; j++) {
				outCostMap[i*w + j] = edges.at<float>(i + borderSize, j + borderSize);
			}
		}
	}

	void ReadEdges(char* fileName, FImage& edge)
	{
		int w = edge.width();
		int h = edge.height();
		FILE* fp = fopen(fileName, "rb");
		int size = fread(edge.pData, sizeof(float), w*h, fp);
		assert(size == w*h);
		fclose(fp);
	}

}

using namespace cv;


Mat2f callRicFlow(const std::string &srcFile, const std::string &tarFile, const std::string &matchFile, const std::string &sedModelFile)
{
	FImage img1, img2;
	FImage matches, costMap;

	img1.imread(srcFile.c_str());
	GetCostMap(srcFile.c_str(), costMap, sedModelFile);
	img2.imread(tarFile.c_str());

	// 	costMap.allocate(img1.width(), img1.height(), 1);
	// 	ReadEdges(argv[3], costMap);

	ReadMatches(matchFile.c_str(), matches);

	int w = img1.width();
	int h = img1.height();
	if (img2.width() != w || img2.height() != h) {
		printf("RIC can only handle images with the same dimension!\n");
		return Mat2f();
	}

	RIC ric;
	FImage u, v;
	ric.SetSuperpixelSize(100);
	ric.Interpolate(img1, img2, costMap, matches, u, v);

	Mat2f flow(h, w);
	int i = 0;
	for(int y=0; y<h; ++y)
		for (int x = 0; x < w; ++x, ++i)
		{
			flow(y, x) = Vec2f(x + u.pData[i], y + v.pData[i]);
		}

	return flow;
	//OpticFlowIO::WriteFlowFile(u.pData, v.pData, w, h, argv[4]);

#if 0
	// save output flow
	char baseName[256], outName[256];

	// get base Name, not support multi-char language, like CJK
	strcpy(baseName, argv[3]);
	char* dot = strrchr(baseName, '.');
	if (dot != NULL) dot[0] = '\0';

	// save the flow and the visualization image
	strcpy(outName, baseName);
	strcat(outName, ".ric.flo");
	OpticFlowIO::WriteFlowFile(u.pData, v.pData, w, h, outName);
	strcpy(outName, baseName);
	strcat(outName, ".ric.png");
	OpticFlowIO::SaveFlowAsImage(outName, u.pData, v.pData, w, h);
#endif
}