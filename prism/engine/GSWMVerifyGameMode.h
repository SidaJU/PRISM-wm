// W2 验证专用 GameMode：自动 Spawn 采集 Actor。
// 目的：不需要在编辑器里手动摆放任何东西（命令行无法操作 GUI）。

#pragma once

#include "CoreMinimal.h"
#include "GameFramework/GameModeBase.h"
#include "GSWMVerifyGameMode.generated.h"

UCLASS()
class AGSWMVerifyGameMode : public AGameModeBase
{
	GENERATED_BODY()

public:
	AGSWMVerifyGameMode();

	virtual void StartPlay() override;
};
