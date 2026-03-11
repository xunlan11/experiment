#include <stdio.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netdb.h>
#include <pthread.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>

#define SERVER_PORT 5432 // 服务器监听端口号
#define MAX_LINE 256     // 缓冲区最大长度

int sock_fd;

// 接收消息线程
void* receive_message(void* arg) 
{
    char buf[MAX_LINE];         
    int len;
    while((len = recv(sock_fd, buf, sizeof(buf), 0)) > 0) 
    {
        buf[len] = '\0'; // 末尾结束符
        printf("receive:%s", buf); 
    }
    return NULL;
}

// 发送消息线程
void* send_message(void* arg) 
{
    char buf[MAX_LINE];     
    while(fgets(buf, sizeof(buf), stdin)) 
    {
        buf[MAX_LINE-1] = '\0';
        int len = strlen(buf) + 1; 
        send(sock_fd, buf, len, 0); 
    }
    return NULL;
}

int main(int argc, char* argv[]) 
{
    FILE *fp;                 
    struct hostent *hp;     // 存储主机信息
    struct sockaddr_in sin; // 服务器地址结构
    char *host;                 
    pthread_t recv_thread, send_thread; // 线程标识符

    // 检查主机名，正确则获取
    if(argc == 2) 
    {
        host = argv[1];         
    } else {
        fprintf(stderr, "wrong host\n");
        exit(1);          
    }

    // 获取主机信息
    hp = gethostbyname(host);
    if(!hp) 
    {
        fprintf(stderr, "unknown host: %s\n", host);
        exit(1);            
    }

    // 初始化服务器地址结构
    bzero((char*)&sin, sizeof(sin));
    sin.sin_family = AF_INET; 
    bcopy(hp->h_addr_list[0], (char*)&sin.sin_addr, hp->h_length);
    sin.sin_port = htons(SERVER_PORT);

    // 创建套接字
    if((sock_fd = socket(PF_INET, SOCK_STREAM, 0)) < 0) 
    {
        perror("socket error\n"); 
        exit(1);                     
    }

    // 连接服务器
    if(connect(sock_fd, (struct sockaddr*)&sin, sizeof(sin)) < 0) 
    {
        perror("connect error\n"); 
        close(sock_fd);        
        exit(1);                 
    }

    // 创建接收和发送线程
    pthread_create(&recv_thread, NULL, receive_message, NULL);
    pthread_create(&send_thread, NULL, send_message, NULL);

    // 等待线程结束
    pthread_join(recv_thread, NULL);
    pthread_join(send_thread, NULL);

    close(sock_fd);
    return 0;
}