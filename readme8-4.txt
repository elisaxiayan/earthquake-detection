如何使用：
在拉下image之后运行：
	docker run -d --name "随便" -p "any_free_port":8000 \
	-e qwen_key="your_qwen_api" -e openai_key="your_openai_api" -e web_key="your_web_api" \
	"image_name"

#这里的“随便”是你给容器的名字，"any_free_port"在上下文是服务器中同一个可用的空端口
#上述三个api第一个就是千问模型api，第二个是openai模型api，第三个是高德web服务api。
#"image_name"就是拉下来的image的名字。

然后即可运行，有两种模式，1为运行单条，2为运行多条。

如下是检测单条：
	curl -s -X POST http://"服务器ip":"any_free_port"/verify \
  	-H "Content-Type: application/json" \
  	-d '{"text":"any_text"}'

#上文中的"any_text"就是待检测文本，"服务器ip"就是部署该容器的服务器的ip

如下是检测多条：
	curl -s -X POST http://"服务器ip":"any_free_port"/verify \
        -H "Content-Type: application/json" \
        -d '{"text":["text1", "text2", "text3"]}'

#基本同上，["text1", "text2", "text3"]list可延长。
