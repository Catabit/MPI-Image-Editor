#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "math.h"
#include <mpi.h>

#define SMOOTH 0
#define BLUR 1
#define SHARPEN 2
#define MEAN 3
#define EMBOSS 4

#define IMAGE_CHUNK_TAG 100
#define TASKSET_TAG 200
#define NEIGHBOUR_ROW_TAG 300

#define FREE
//#define DEBUG


typedef struct
{
    int img_type; //5/6
    int maxval;
    int width, height;
    unsigned char** raster;
} image;

typedef struct 
{
    int img_type;
    int width, height;
    unsigned char* raster_vector;
} image_chunk;

typedef struct 
{
    char numTasks;
    char *tasks;
} taskSet;


MPI_Datatype mpi_taskSet;
MPI_Datatype mpi_image_chunk;

void readInput(const char * fileName, image *img) {

    FILE *in = fopen(fileName, "rb");
    char buff[255];

    fgets(buff, 10, in); //type line P5/P6
    img->img_type = atoi(buff+1);

    fgets(buff, 255, in); //width height
    img->width = atoi(buff); // up to space
    strcpy(buff, strchr(buff, ' ')); //remove up to space+space
    img->height = atoi(buff);

    fgets(buff, 255, in);
    img->maxval = atoi(buff);

    unsigned char** raster;

    if (img->img_type == 5) { //bw

        raster = malloc(img->height * sizeof(char*));
        int i;

        for (i=0; i<img->height; i++) {
            raster[i] = malloc(sizeof(char) * img->width );
            fread(raster[i], 1, img->width, in);
        }
    }
    if (img->img_type == 6) { //color

        raster = malloc(img->height * sizeof(char*));
        int i;

        for (i=0; i<img->height; i++) {
            raster[i] = malloc(sizeof(char) * img->width *3);
            fread(raster[i], 1, img->width*3, in);
        }
    }

    img->raster=raster;
    fclose(in);
}

void writeData(const char * fileName, image *img) {
    FILE *out = fopen(fileName, "wb");
    if (img->img_type == 5) {
        fprintf(out, "P%d\n%d %d\n%d\n", img->img_type, img->width, img->height, img->maxval);
    } else if (img->img_type == 6) {
        fprintf(out, "P%d\n%d %d\n%d\n", img->img_type, img->width/3, img->height, img->maxval);    
    }
    
    int i=0;
    for(i=0; i<img->height; i++) {
        if (img->img_type == 5)
            fwrite(img->raster[i], 1, img->width, out);
        if (img->img_type == 6)
            fwrite(img->raster[i], 1, img->width, out);
        free(img->raster[i]);
    }
    fclose(out);
    free(img->raster);
}

int stringToTask(char* str){
    if (!strcmp(str, "smooth"))
        return SMOOTH;
    if (!strcmp(str, "blur"))
        return BLUR;
    if (!strcmp(str, "sharpen"))
        return SHARPEN;
    if (!strcmp(str, "mean"))
        return MEAN;
    if (!strcmp(str, "emboss"))
        return EMBOSS;
    return -1;
}

void image_chunkMPIinit(){
    const int nitems=3;
    int          blocklengths[3] = {1,1, 1};
    MPI_Datatype types[3] = {MPI_INT, MPI_INT, MPI_INT};
    MPI_Aint     offsets[3];

    offsets[0] = offsetof(image_chunk, img_type);
    offsets[1] = offsetof(image_chunk, width);
    offsets[2] = offsetof(image_chunk, height);

    MPI_Type_create_struct(nitems, blocklengths, offsets, types, &mpi_image_chunk);
    MPI_Type_commit(&mpi_image_chunk);
}

void taskSetMPIinit(){
    const int nitems=1;
    int          blocklengths[1] = {1};
    MPI_Datatype types[1] = {MPI_CHAR};
    MPI_Aint     offsets[1];

    offsets[0] = offsetof(taskSet, numTasks);

    MPI_Type_create_struct(nitems, blocklengths, offsets, types, &mpi_taskSet);
    MPI_Type_commit(&mpi_taskSet);
}

void mpi_init_structs(){
    image_chunkMPIinit();
    taskSetMPIinit();
}

int mpi_SendtaskSet(taskSet *t, int reiceiver, MPI_Comm comm) {
    
    int result;
    result = MPI_Send(t, 1, mpi_taskSet, reiceiver, TASKSET_TAG, comm);
    if (result!=0)
        return result;
    result = MPI_Send(t->tasks, t->numTasks, MPI_CHAR, reiceiver, TASKSET_TAG+1, comm);

    return result;
}

void mpi_RecvtaskSet(taskSet *t, MPI_Comm comm) {
    MPI_Recv(t, 1, mpi_taskSet, 0, TASKSET_TAG, comm, MPI_STATUS_IGNORE);

    t->tasks = malloc(t->numTasks*sizeof(char));
    MPI_Recv(t->tasks, t->numTasks, MPI_CHAR, 0, TASKSET_TAG+1, comm, MPI_STATUS_IGNORE);
}


int mpi_Sendimage_chunk(image_chunk *chunk, int reiceiver, MPI_Comm comm) {
    int result;
    result = MPI_Send(chunk, 1, mpi_image_chunk, reiceiver, IMAGE_CHUNK_TAG, comm);
    if (result!=0)
        return result;
    result = MPI_Send(chunk->raster_vector, (chunk->height)*(chunk->width), MPI_CHAR, reiceiver, IMAGE_CHUNK_TAG+1, comm);

    return result;
}

void mpi_Recvimage_chunk(image_chunk *chunk, int sender, MPI_Comm comm) {
   
    MPI_Recv(chunk, 1, mpi_image_chunk, sender, IMAGE_CHUNK_TAG, comm, MPI_STATUS_IGNORE);

    chunk->raster_vector = malloc((chunk->height)*(chunk->width)*sizeof(char));
    MPI_Recv(chunk->raster_vector, (chunk->height)*(chunk->width), MPI_CHAR, sender, IMAGE_CHUNK_TAG+1, comm, MPI_STATUS_IGNORE);
}

void applyFilter(image_chunk *chunk, char filter, unsigned char* botRow, unsigned char* topRow, unsigned char* result, int rank, int nProcesses) {

    int i, j;
    int step = chunk->img_type==6?3:1;
    int end = rank==nProcesses-1?chunk->height-1:chunk->height; //if running singlethreaded, stop at last line
    int start = rank==0?1:0;

    for (i=start; i<end; i++) {
        for (j=step; j<chunk->width-step; j+=step) { 
            float f[3][3];
            
            switch(filter) 
            {
                case SMOOTH:{
                    f[0][0] = (float)1/9; f[0][1] = (float)1/9; f[0][2] = (float)1/9;
                    f[1][0] = (float)1/9; f[1][1] = (float)1/9; f[1][2] = (float)1/9;
                    f[2][0] = (float)1/9; f[2][1] = (float)1/9; f[2][2] = (float)1/9;
                    break;
                }
                case BLUR:{

                    f[0][0] = (float)1/16; f[0][1] = (float)2/16; f[0][2] = (float)1/16;
                    f[1][0] = (float)2/16; f[1][1] = (float)4/16; f[1][2] = (float)2/16;
                    f[2][0] = (float)1/16; f[2][1] = (float)2/16; f[2][2] = (float)1/16;
                    break;
                }
                case SHARPEN:{
                    f[0][0] = (float)0; f[0][1] = (float)-2/3; f[0][2] = (float)0;
                    f[1][0] = (float)-2/3; f[1][1] = (float)11/3; f[1][2] = (float)-2/3;
                    f[2][0] = (float)0; f[2][1] = (float)-2/3; f[2][2] = (float)0;
                    break;
                }
                case MEAN:{
                    f[0][0] = (float)-1; f[0][1] = (float)-1; f[0][2] = (float)-1;
                    f[1][0] = (float)-1; f[1][1] = (float)9; f[1][2] = (float)-1;
                    f[2][0] = (float)-1; f[2][1] = (float)-1; f[2][2] = (float)-1;
                    break;
                }
                case EMBOSS:{
                    f[0][0] = (float)0; f[0][1] = (float)1; f[0][2] = (float)0;
                    f[1][0] = (float)0; f[1][1] = (float)0; f[1][2] = (float)0;
                    f[2][0] = (float)0; f[2][1] = (float)-1; f[2][2] = (float)0;
                    break;
                }
                default: {
                    f[0][0] = (float)0; f[0][1] = (float)0; f[0][2] = (float)0;
                    f[1][0] = (float)0; f[1][1] = (float)0; f[1][2] = (float)0;
                    f[2][0] = (float)0; f[2][1] = (float)0; f[2][2] = (float)0;
                    break;
                }
            }

            if (chunk->img_type==5) {
                float sum = 0;

                if (i==0) { //use toprow
                    sum += topRow[j-1]*f[0][0] + topRow[j]*f[0][1] + topRow[j+1]*f[0][2] +
                    chunk->raster_vector[(i)*chunk->width+j-1]*f[1][0] + chunk->raster_vector[(i)*chunk->width+j]*f[1][1] + chunk->raster_vector[(i)*chunk->width+j+1]*f[1][2] +
                    chunk->raster_vector[(i+1)*chunk->width+j-1]*f[2][0] + chunk->raster_vector[(i+1)*chunk->width+j]*f[2][1] + chunk->raster_vector[(i+1)*chunk->width+j+1]*f[2][2];
                } else if (i==chunk->height-1) { //use botrow
                    sum += chunk->raster_vector[(i-1)*chunk->width+j-1]*f[0][0] + chunk->raster_vector[(i-1)*chunk->width+j]*f[0][1] + chunk->raster_vector[(i-1)*chunk->width+j+1]*f[0][2] +
                    chunk->raster_vector[(i)*chunk->width+j-1]*f[1][0] + chunk->raster_vector[(i)*chunk->width+j]*f[1][1] + chunk->raster_vector[(i)*chunk->width+j+1]*f[1][2] +
                    botRow[j-1]*f[2][0] + botRow[j]*f[2][1] + botRow[j+1]*f[2][2];
                } else {
                    
                    sum += chunk->raster_vector[(i-1)*chunk->width+j-1]*f[0][0] + chunk->raster_vector[(i-1)*chunk->width+j]*f[0][1] + chunk->raster_vector[(i-1)*chunk->width+j+1]*f[0][2] +
                    chunk->raster_vector[(i)*chunk->width+j-1]*f[1][0] + chunk->raster_vector[(i)*chunk->width+j]*f[1][1] + chunk->raster_vector[(i)*chunk->width+j+1]*f[1][2] +
                    chunk->raster_vector[(i+1)*chunk->width+j-1]*f[2][0] + chunk->raster_vector[(i+1)*chunk->width+j]*f[2][1] + chunk->raster_vector[(i+1)*chunk->width+j+1]*f[2][2];
                }
                
                result[i*chunk->width + j] = (unsigned char) sum;
            } else if (chunk->img_type==6) {
                float sumr =0, sumg=0, sumb=0;

                if (i==0) { //use toprow
                    sumr += topRow[j-3]*f[0][0] + topRow[j]*f[0][1] + topRow[j+3]*f[0][2] +
                    chunk->raster_vector[(i)*chunk->width+j-3]*f[1][0] + chunk->raster_vector[(i)*chunk->width+j]*f[1][1] + chunk->raster_vector[(i)*chunk->width+j+3]*f[1][2] +
                    chunk->raster_vector[(i+1)*chunk->width+j-3]*f[2][0] + chunk->raster_vector[(i+1)*chunk->width+j]*f[2][1] + chunk->raster_vector[(i+1)*chunk->width+j+3]*f[2][2];

                    sumg += topRow[j-3+1]*f[0][0] + topRow[j+1]*f[0][1] + topRow[j+3+1]*f[0][2] +
                    chunk->raster_vector[(i)*chunk->width+j-3+1]*f[1][0] + chunk->raster_vector[(i)*chunk->width+j+1]*f[1][1] + chunk->raster_vector[(i)*chunk->width+j+3+1]*f[1][2] +
                    chunk->raster_vector[(i+1)*chunk->width+j-3+1]*f[2][0] + chunk->raster_vector[(i+1)*chunk->width+j+1]*f[2][1] + chunk->raster_vector[(i+1)*chunk->width+j+3+1]*f[2][2];

                    sumb += topRow[j-3+2]*f[0][0] + topRow[j+2]*f[0][1] + topRow[j+3+2]*f[0][2] +
                    chunk->raster_vector[(i)*chunk->width+j-3+2]*f[1][0] + chunk->raster_vector[(i)*chunk->width+j+2]*f[1][1] + chunk->raster_vector[(i)*chunk->width+j+3+2]*f[1][2] +
                    chunk->raster_vector[(i+1)*chunk->width+j-3+2]*f[2][0] + chunk->raster_vector[(i+1)*chunk->width+j+2]*f[2][1] + chunk->raster_vector[(i+1)*chunk->width+j+3+2]*f[2][2];

                } else if (i==chunk->height-1) { //use botrow
                    sumr += chunk->raster_vector[(i-1)*chunk->width+j-3]*f[0][0] + chunk->raster_vector[(i-1)*chunk->width+j]*f[0][1] + chunk->raster_vector[(i-1)*chunk->width+j+3]*f[0][2] +
                    chunk->raster_vector[(i)*chunk->width+j-3]*f[1][0] + chunk->raster_vector[(i)*chunk->width+j]*f[1][1] + chunk->raster_vector[(i)*chunk->width+j+3]*f[1][2] +
                    botRow[j-3]*f[2][0] + botRow[j]*f[2][1] + botRow[j+3]*f[2][2];

                    sumg += chunk->raster_vector[(i-1)*chunk->width+j-3+1]*f[0][0] + chunk->raster_vector[(i-1)*chunk->width+j+1]*f[0][1] + chunk->raster_vector[(i-1)*chunk->width+j+3+1]*f[0][2] +
                    chunk->raster_vector[(i)*chunk->width+j-3+1]*f[1][0] + chunk->raster_vector[(i)*chunk->width+j+1]*f[1][1] + chunk->raster_vector[(i)*chunk->width+j+3+1]*f[1][2] +
                    botRow[j-3+1]*f[2][0] + botRow[j+1]*f[2][1] + botRow[j+3+1]*f[2][2];

                    sumb += chunk->raster_vector[(i-1)*chunk->width+j-3+2]*f[0][0] + chunk->raster_vector[(i-1)*chunk->width+j+2]*f[0][1] + chunk->raster_vector[(i-1)*chunk->width+j+3+2]*f[0][2] +
                    chunk->raster_vector[(i)*chunk->width+j-3+2]*f[1][0] + chunk->raster_vector[(i)*chunk->width+j+2]*f[1][1] + chunk->raster_vector[(i)*chunk->width+j+3+2]*f[1][2] +
                    botRow[j-3+2]*f[2][0] + botRow[j+2]*f[2][1] + botRow[j+3+2]*f[2][2];

                } else {
                    sumr += chunk->raster_vector[(i-1)*chunk->width+j-3]*f[0][0] + chunk->raster_vector[(i-1)*chunk->width+j]*f[0][1] + chunk->raster_vector[(i-1)*chunk->width+j+3]*f[0][2] +
                    chunk->raster_vector[(i)*chunk->width+j-3]*f[1][0] + chunk->raster_vector[(i)*chunk->width+j]*f[1][1] + chunk->raster_vector[(i)*chunk->width+j+3]*f[1][2] +
                    chunk->raster_vector[(i+1)*chunk->width+j-3]*f[2][0] + chunk->raster_vector[(i+1)*chunk->width+j]*f[2][1] + chunk->raster_vector[(i+1)*chunk->width+j+3]*f[2][2];

                    sumg += chunk->raster_vector[(i-1)*chunk->width+j-3+1]*f[0][0] + chunk->raster_vector[(i-1)*chunk->width+j+1]*f[0][1] + chunk->raster_vector[(i-1)*chunk->width+j+3+1]*f[0][2] +
                    chunk->raster_vector[(i)*chunk->width+j-3+1]*f[1][0] + chunk->raster_vector[(i)*chunk->width+j+1]*f[1][1] + chunk->raster_vector[(i)*chunk->width+j+3+1]*f[1][2] +
                    chunk->raster_vector[(i+1)*chunk->width+j-3+1]*f[2][0] + chunk->raster_vector[(i+1)*chunk->width+j+1]*f[2][1] + chunk->raster_vector[(i+1)*chunk->width+j+3+1]*f[2][2];

                    sumb += chunk->raster_vector[(i-1)*chunk->width+j-3+2]*f[0][0] + chunk->raster_vector[(i-1)*chunk->width+j+2]*f[0][1] + chunk->raster_vector[(i-1)*chunk->width+j+3+2]*f[0][2] +
                    chunk->raster_vector[(i)*chunk->width+j-3+2]*f[1][0] + chunk->raster_vector[(i)*chunk->width+j+2]*f[1][1] + chunk->raster_vector[(i)*chunk->width+j+3+2]*f[1][2] +
                    chunk->raster_vector[(i+1)*chunk->width+j-3+2]*f[2][0] + chunk->raster_vector[(i+1)*chunk->width+j+2]*f[2][1] + chunk->raster_vector[(i+1)*chunk->width+j+3+2]*f[2][2];
                }

                result[i*chunk->width + j] = (unsigned char) sumr;
                result[i*chunk->width + j + 1] = (unsigned char) sumg;
                result[i*chunk->width + j + 2] = (unsigned char) sumb;
            }
        }
    }

}



int main(int argc, char * argv[]) {

    int rank;
    int nProcesses;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nProcesses);

    mpi_init_structs();

    if (rank == 0) {
        image input, output;
        image_chunk *firstChunk;
        //read the input image file
        readInput(argv[1], &input);

        // Read the filters and construct the taskset
        char *tasks = malloc((argc-3)*sizeof(char));
        int i; 
        for (i=3; i<argc; i++) {
            tasks[i-3] = stringToTask(argv[i]);
        }
        taskSet *t;
        t = malloc(sizeof(taskSet));
        t->numTasks = argc-3;
        t->tasks = tasks;

        //send the taskset
        for (i=1; i<nProcesses; i++) {
            mpi_SendtaskSet(
                t, i, MPI_COMM_WORLD);
        }

        MPI_Barrier(MPI_COMM_WORLD);

        //set the image chunks

        int sectorSize = input.height / nProcesses;
        for (i=0; i<nProcesses; i++){
            int startLine = i*sectorSize;
            int endLine = (i+1)*sectorSize;
            if (i==nProcesses-1)
                endLine = input.height; 
            image_chunk *c;
            c = malloc(sizeof(image_chunk));
            c->img_type = input.img_type;
            c->height=endLine-startLine;
            c->width = input.width;



            if (input.img_type==6){
                c->width*=3;
            }
            c->raster_vector = malloc((c->width)*(c->height)*sizeof(char));
            int k, o;

            for (k=startLine, o=0; k<endLine; k++, o++)
                memcpy(c->raster_vector+(o*c->width), input.raster[k], c->width);
            if (i==0) {
                firstChunk = c;
            } else {
                mpi_Sendimage_chunk(c, i, MPI_COMM_WORLD);
                #ifdef FREE
                free(c->raster_vector);
                free(c);
                #endif
            }

            
        }

        MPI_Barrier(MPI_COMM_WORLD);
        
        //work on first chunk
        int filter = 0;
        unsigned char *botRow = malloc(firstChunk->width*sizeof(char));
        for (filter=0; filter<t->numTasks; filter++) {
            if (nProcesses>1) {
                MPI_Recv(botRow, firstChunk->width, MPI_CHAR, rank+1, NEIGHBOUR_ROW_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Send(firstChunk->raster_vector+(firstChunk->height-1)*firstChunk->width, firstChunk->width, MPI_CHAR, rank+1, NEIGHBOUR_ROW_TAG, MPI_COMM_WORLD);
                
            }
            MPI_Barrier(MPI_COMM_WORLD); 

            
            unsigned char* result = malloc(firstChunk->width*firstChunk->height*sizeof(char));
            memcpy(result, firstChunk->raster_vector, firstChunk->width*firstChunk->height);

            applyFilter(firstChunk, t->tasks[filter], botRow, NULL, result, rank, nProcesses);

            #ifdef FREE
            free(firstChunk->raster_vector);
            #endif
            firstChunk->raster_vector=result;
            
            MPI_Barrier(MPI_COMM_WORLD);       
        }


        //wait for all the chunks back
        //assemble output image

        output.height = input.height;
        output.width = input.width;
        output.maxval = input.maxval;
        output.img_type = input.img_type;

        if (output.img_type == 6) {
            output.width*=3;
        }

        output.raster = malloc(output.height * sizeof(char*));
        for (i=0; i<output.height; i++) {
            output.raster[i] = malloc(output.width*sizeof(char));
        }

        int buildLine = 0;
        int k, o;

        for (k=buildLine, o=0; k<buildLine+firstChunk->height; k++, o++){
            memcpy(output.raster[k], firstChunk->raster_vector+(o*firstChunk->width), firstChunk->width);
        }
        buildLine+=firstChunk->height;

        image_chunk *c;
        for (i=1; i<nProcesses; i++){

            
            c = malloc(sizeof(image_chunk));
            
            mpi_Recvimage_chunk(c, i, MPI_COMM_WORLD);
            
            int k, o;
            for (k=buildLine, o=0; k<buildLine+c->height; k++, o++){
                memcpy(output.raster[k], c->raster_vector+(o*c->width), c->width);
            }
            buildLine+=c->height;

            #ifdef FREE
            free(c->raster_vector);
            free(c);
            #endif
        }

        
        //write the output image file
        writeData(argv[2], &output);
      
        #ifdef FREE
        free(t->tasks);
        free(t);
        free(firstChunk->raster_vector);
        free(firstChunk);
        #endif


    } else {
        //get taskSet        
        taskSet *t = malloc(sizeof(taskSet));
        mpi_RecvtaskSet(t, MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);

        //get image chunk
        //image_chunk *chunk = malloc(sizeof(image_chunk));
        image_chunk *chunk;
        chunk = malloc(sizeof(image_chunk));
  
        mpi_Recvimage_chunk(chunk, 0, MPI_COMM_WORLD);

        MPI_Barrier(MPI_COMM_WORLD);

        int filter = 0;
        unsigned char *topRow, *botRow;
        topRow = malloc(chunk->width*sizeof(char));
        botRow = malloc(chunk->width*sizeof(char));
        
        for (filter = 0; filter<t->numTasks; filter++) {
            //get/send top+bottom line to neighbors
            if (rank>0) { //needs to send top row, receive a new top row  
                if (rank%2==1) {
                    MPI_Send(chunk->raster_vector, chunk->width, MPI_CHAR, rank-1, NEIGHBOUR_ROW_TAG, MPI_COMM_WORLD);
                    MPI_Recv(topRow, chunk->width, MPI_CHAR, rank-1, NEIGHBOUR_ROW_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                } else {
                    MPI_Recv(topRow, chunk->width, MPI_CHAR, rank-1, NEIGHBOUR_ROW_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    MPI_Send(chunk->raster_vector, chunk->width, MPI_CHAR, rank-1, NEIGHBOUR_ROW_TAG, MPI_COMM_WORLD);
                }
            }
            if (rank<nProcesses-1) { //needs to send bottom row, receive a new bottom row
                
                if (rank%2) {
                    MPI_Send(chunk->raster_vector+((chunk->height-1)*chunk->width), chunk->width, MPI_CHAR, rank+1, NEIGHBOUR_ROW_TAG, MPI_COMM_WORLD);
                    MPI_Recv(botRow, chunk->width, MPI_CHAR, rank+1, NEIGHBOUR_ROW_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                } else {
                    MPI_Recv(botRow, chunk->width, MPI_CHAR, rank+1, NEIGHBOUR_ROW_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    MPI_Send(chunk->raster_vector+((chunk->height-1)*chunk->width), chunk->width, MPI_CHAR, rank+1, NEIGHBOUR_ROW_TAG, MPI_COMM_WORLD);
                }
            }

            MPI_Barrier(MPI_COMM_WORLD); 

        //start processing first filter

            unsigned char* result = malloc(chunk->width*chunk->height*sizeof(char));
            memcpy(result, chunk->raster_vector, chunk->width*chunk->height);

            applyFilter(chunk, t->tasks[filter], botRow, topRow, result, rank, nProcesses);

            #ifdef FREE
            free(chunk->raster_vector);
            #endif
            chunk->raster_vector=result;


        //process chunk (barrier after each filter), goto start processing
            MPI_Barrier(MPI_COMM_WORLD); 
        }

        //send final chunk back to 0

        mpi_Sendimage_chunk(chunk, 0, MPI_COMM_WORLD);

        #ifdef FREE
        free(t->tasks);
        free(t);
        free(botRow);
        free(topRow);
        free(chunk->raster_vector);
        free(chunk);
        #endif
    }

    #ifdef FREE
    MPI_Type_free(&mpi_image_chunk);
    MPI_Type_free(&mpi_taskSet);
    #endif
    
    MPI_Finalize();

    return 0;
}
