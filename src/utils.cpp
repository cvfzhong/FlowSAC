 
#include"utils.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
using namespace std;
using namespace cv;

#include"BFC/err.h"
#include"BFC/portable.h"
#include"BFC/stdf.h"

// first four bytes, should be the same in little endian
#define TAG_FLOAT 202021.25  // check for this when READING the file
#define TAG_STRING "PIEH"    // use this when WRITING the file

struct CError : public exception
{
	CError(const char* msg) { strcpy(message, msg); }
	CError(const char* fmt, int d) { sprintf(message, fmt, d); }
	CError(const char* fmt, float f) { sprintf(message, fmt, f); }
	CError(const char* fmt, const char *s) { sprintf(message, fmt, s); }
	CError(const char* fmt, const char *s,
		int d) {
		sprintf(message, fmt, s, d);
	}
	char message[1024];         // longest allowable message
};

Mat2f readFlowFile(const std::string &_filename)
{
	const char *filename = _filename.c_str();

	if (filename == NULL)
		throw CError("ReadFlowFile: empty filename");

	const char *dot = strrchr(filename, '.');
	if (strcmp(dot, ".flo") != 0) {
		printf("%s\n", filename);
		throw CError("ReadFlowFile (%s): extension .flo expected", filename);
	}

	FILE *stream = fopen(filename, "rb");
	if (stream == 0)
		throw CError("ReadFlowFile: could not open %s", filename);

	int width, height;
	float tag;

	if ((int)fread(&tag, sizeof(float), 1, stream) != 1 ||
		(int)fread(&width, sizeof(int), 1, stream) != 1 ||
		(int)fread(&height, sizeof(int), 1, stream) != 1)
		throw CError("ReadFlowFile: problem reading file %s", filename);

	if (tag != TAG_FLOAT) // simple test for correct endian-ness
		throw CError("ReadFlowFile(%s): wrong tag (possibly due to big-endian machine?)", filename);

	// another sanity check to see that integers were read correctly (99999 should do the trick...)
	if (width < 1 || width > 99999)
		throw CError("ReadFlowFile(%s): illegal width %d", filename, width);

	if (height < 1 || height > 99999)
		throw CError("ReadFlowFile(%s): illegal height %d", filename, height);

	int nBands = 2;
	//CShape sh(width, height, nBands);
	//img.ReAllocate(sh);

	Mat2f w(height, width);// = Mat::zeros(height, width, CV_32FC2);

						   //printf("reading %d x %d x 2 = %d floats\n", width, height, width*height*2);
	int n = nBands * width;
	for (int y = 0; y < height; y++) {
		//float* ptr = &img.Pixel(0, y, 0);
		float* ptr = (float*)w.ptr(y);
		if ((int)fread(ptr, sizeof(float), n, stream) != n)
			throw CError("ReadFlowFile(%s): file is too short", filename);
	}
	if (fgetc(stream) != EOF)
		throw CError("ReadFlowFile(%s): file is too long", filename);

	fclose(stream);

	offset2xy(w);

	return w;
}

void writeFlowFile(const Mat2f &uv, const std::string &_filename)
{
	const char *filename = _filename.c_str();

	Mat2f w;
	xy2offset(uv, w);

	if (filename == NULL)
		throw CError("WriteFlowFile: empty filename");

	const char *dot = strrchr(filename, '.');
	if (dot == NULL)
		throw CError("WriteFlowFile: extension required in filename '%s'", filename);

	if (strcmp(dot, ".flo") != 0)
		throw CError("WriteFlowFile: filename '%s' should have extension '.flo'", filename);

	//CShape sh = img.Shape();
	int width = w.cols, height = w.rows, nBands = w.channels();

	if (nBands != 2)
		throw CError("WriteFlowFile(%s): image must have 2 bands", filename);

	FILE *stream = fopen(filename, "wb");
	if (stream == 0)
		throw CError("WriteFlowFile: could not open %s", filename);

	// write the header
	fprintf(stream, TAG_STRING);
	if ((int)fwrite(&width, sizeof(int), 1, stream) != 1 ||
		(int)fwrite(&height, sizeof(int), 1, stream) != 1)
		throw CError("WriteFlowFile(%s): problem writing header", filename);

	// write the rows
	int n = nBands * width;
	for (int y = 0; y < height; y++) {
		float* ptr = (float*)w.ptr(y);
		if ((int)fwrite(ptr, sizeof(float), n, stream) != n)
			throw CError("WriteFlowFile(%s): problem writing data", filename);
	}

	fclose(stream);
}

//#ifndef _WIN64
#if 1
#define PNG_NO_READ_EXPAND
#define _USE_MATH_DEFINES
#include"KITTI/io_flow.h"

void readFlowKitti(Mat2f &uv, Mat1b &mask, const string& gtname)
{
	FlowImage gt(gtname);
	int cols = gt.width_;
	int rows = gt.height_;
	float *data = gt.data();

	uv = Mat2f::zeros(rows, cols);
	mask = Mat1b::zeros(rows, cols);

	for (int i = 0; i<rows; i++) {
		for (int j = 0; j<cols; j++) {
			uv(i, j) = Vec2f(data[3 * (i*cols + j) + 0],data[3 * (i*cols + j) + 1]);
			mask(i, j) = (uchar)(data[3 * (i*cols + j) + 2]+0.5);
		}
	}
	offset2xy(uv);
}
void writeFlowKitti(Mat2f &uv, const string& floname) 
{
	int width_ = uv.cols;
	int height_ = uv.rows;

	int rows = uv.rows;
	int cols = uv.cols;

	float *data = (float*)malloc(width_*height_ * 3 * sizeof(float));

	for (int i = 0; i<rows; i++) {
		for (int j = 0; j<cols; j++) {
			data[3 * (i*cols + j) + 0] = uv(i, j)[0];
			data[3 * (i*cols + j) + 1] = uv(i, j)[1];
			data[3 * (i*cols + j) + 2] = 1;
		}
	}

	FlowImage fim(data, width_, height_);
	fim.write(floname);
}
#else
void readFlowKitti(Mat2f &uv, Mat1b &mask, const string& gtname)
{
	throw "unimplemented";
}
void writeFlowKitti(Mat2f &uv, const string& floname)
{
	throw "unimplemented";
}
#endif

int readFlowX(const std::string &flowFile, Mat2f &flow, Mat1b &mask)
{
	std::string ext = flowFile.substr(flowFile.find_last_of('.'));
	int r = 0;
	if (stricmp(ext.c_str(),".png")==0)
	{
		readFlowKitti(flow, mask, flowFile);
		r = 2;
	}
	else 
	{
		flow = readFlowFile(flowFile);
		mask.create(flow.size());
		mask = 1;
		r = 1;
	}
	
	return r;
}


void  saveMatches(const std::vector<Match> &matches, const std::string &_filename)
{
	FILE *fp = fopen(_filename.c_str(), "w");
	for (auto &m : matches)
	{
		fprintf(fp, "%g %g %g %g\n", m.x0, m.y0, m.x1, m.y1);
	}
	fclose(fp);
}

void  readMatches(std::vector<Match> &matches, const std::string &file)
{
	FILE *fp = fopen(file.c_str(), "r");
	if (!fp)
		FF_EXCEPTION(ERR_FILE_OPEN_FAILED, file.c_str());

	std::vector<Match> vm;
	vm.reserve(2048);

	Match m;
	char buf[1024];
	float score = 0;
	while (fgets(buf, sizeof(buf), fp))
	{
		if (sscanf(buf, "%f %f %f %f %f", &m.x0, &m.y0, &m.x1, &m.y1, &score) >= 4)
			vm.push_back(m);
	}

	vm.swap(matches);
	fclose(fp);
}

//void  saveMatchesEX(const std::vector<Match> &matches, const std::string &_filename)
//{
//	FILE *fp = fopen(_filename.c_str(), "w");
//	for (auto &m : matches)
//	{
//		fprintf(fp, "%d %d %g\n", m.id, m.label, m.score);
//	}
//	fclose(fp);
//}

//void  readMatchesEX(std::vector<Match> &matches, const std::string &file)
//{
//	FILE *fp = fopen(file.c_str(), "r");
//	if (!fp)
//		FF_EXCEPTION(ERR_FILE_OPEN_FAILED, file.c_str());
//
//	Match m;
//	char buf[1024];
//	while (fgets(buf, sizeof(buf), fp))
//	{
//		if (sscanf(buf, "%d %d %g", &m.id, &m.label, &m.score) >= 3)
//		{
//			if ((uint)m.id >= matches.size())
//				FF_EXCEPTION(ERR_INVALID_OP, "");
//			auto &mx(matches[m.id]);
//			mx.label = m.label;
//			mx.score = m.score;
//		}
//	}
//	
//	fclose(fp);
//}

Mat1f getFlowError(const Mat2f &flow, const Mat2f &gt)
{
	Mat1f err(flow.size());
	for_each_3(DWHNC(flow), DNC(gt), DN1(err), [](const float *m, const float *g, float &e) {
		float dx = m[0] - g[0], dy = m[1] - g[1];
		e = sqrt(dx*dx + dy*dy);
	});
	return err;
}
 
double getAEE(const Mat2f &gt, const Mat2f &flow, Mat1b mask) 
{
	double aee = 0;
	if (mask.empty())
	{
		for_each_2(DWHN2(gt), DN2(flow), [&aee](const Vec2f &g, const Vec2f &f) {
			Vec2f d = f - g;
			aee += sqrt(d.dot(d));
		});
		aee /= gt.rows*gt.cols;
	}
	else
	{
		int n = 0;
		for_each_3(DWHN2(gt), DN2(flow), DN1(mask), [&aee,&n](const Vec2f &g, const Vec2f &f, uchar m) {
			if (m)
			{
				Vec2f d = f - g;
				aee += sqrt(d.dot(d));
				++n;
			}
		});
		if (n > 0)
			aee /= n;
	}

	return aee;
}

void crossCheck(const Mat2f &fflow, const Mat2f &bflow, std::vector<Match> &vMatches, float dT, int ds)
{
	vMatches.clear();
	dT *= dT;

	for_each_1c(DWHNC(fflow), [&bflow, dT, ds, &vMatches](const float *p, int x, int y) {
		if (x % ds == 0 && y % ds == 0)
		{
			int tx = int(p[0] + 0.5), ty = int(p[1] + 0.5);
			if (uint(tx) < bflow.cols && uint(ty) < bflow.rows)
			{
				auto sm = bflow(ty, tx);
				float dx = sm[0] - x, dy = sm[1] - y;
				float d = dx*dx + dy*dy;
				if (d < dT)
				{
					Match m;
					m.x0 = x, m.y0 = y;
					m.x1 = p[0], m.y1 = p[1];
					vMatches.push_back(m);
				}
			}
		}
	});
}

void saveFlowList(const std::string &file, const std::vector<FlowPair> &pairs)
{
	FILE *fp = fopen(file.c_str(), "w");
	if (!fp)
		FF_EXCEPTION(ERR_FILE_OPEN_FAILED, file.c_str());

	for (auto &f : pairs)
	{
		fprintf(fp, "%s\t\t%s\n", f.src.c_str(), f.tar.c_str());
	}
	fclose(fp);
}

void readFlowList(const std::string &file, std::vector<FlowPair> &pairs)
{
	FILE *fp = fopen(file.c_str(), "r");
	if (!fp)
		FF_EXCEPTION(ERR_FILE_OPEN_FAILED, file.c_str());

	pairs.clear();

	char line[1024];
	char src[256], tar[256];
	FlowPair pair;
	//while (fscanf(fp, "%s %s", src, tar) == 2)
	while(fgets(line,sizeof(line),fp) && sscanf(line,"%s %s",src,tar)==2)
	{
		pair.src = src; 
		pair.tar = tar;
		pair.info = line;
		pairs.push_back(pair);
	}

	fclose(fp);
}

Mat1f readEdgeFile(const std::string &filename, Size size) 
{
	Mat1f edge(size);
	FILE *fid = fopen(filename.c_str(), "rb");
	if (!fid)
		FF_EXCEPTION(ERR_FILE_OPEN_FAILED, "");

	fread(edge.data, sizeof(float), size.width*size.height, fid);
	fclose(fid);
	return edge;
}

void saveEdgeFille(const std::string &filename, const Mat1f &_edge)
{
	FILE *fid = fopen(filename.c_str(), "wb");
	if (!fid)
		FF_EXCEPTION(ERR_FILE_OPEN_FAILED, "");

	Mat1f edge = _edge;
	if (_edge.step != sizeof(float)*edge.cols)
		edge = _edge.clone();

	CV_Assert(edge.step == sizeof(float)*edge.cols);

	fwrite(edge.data, sizeof(float), edge.rows * edge.cols, fid);
	fclose(fid);
}







