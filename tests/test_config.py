"""
配置模块单元测试

验证 Settings 能正确从环境变量加载，并返回预期的默认值。
"""

import pytest

from app.config.settings import Settings, get_settings


class TestSettings:
    """Settings 配置类测试"""

    def test_default_values(self, monkeypatch):
        """测试默认配置值是否正确（绕过 .env 和环境变量干扰）"""
        for var in [
            "APP_PORT",
            "PDF_SYNC_PAGE_THRESHOLD",
            "RETRIEVAL_TOP_K",
            "RETRIEVAL_MIN_SCORE",
        ]:
            monkeypatch.delenv(var, raising=False)
        settings = Settings(_env_file=None)
        assert settings.APP_PORT == 8000
        assert settings.PDF_SYNC_PAGE_THRESHOLD == 10
        assert settings.RETRIEVAL_TOP_K == 5
        assert settings.RETRIEVAL_MIN_SCORE == 0.5

    def test_env_override(self, monkeypatch):
        """测试环境变量能正确覆盖默认值"""
        monkeypatch.setenv("APP_PORT", "9000")
        monkeypatch.setenv("PDF_SYNC_PAGE_THRESHOLD", "20")
        settings = Settings()
        assert settings.APP_PORT == 9000
        assert settings.PDF_SYNC_PAGE_THRESHOLD == 20

    def test_path_properties(self):
        """测试 Path 类型属性是否正确"""
        settings = Settings()
        assert settings.chroma_persist_path.name == "chroma_db"
        assert settings.log_dir_path.name == "logs"

    def test_get_settings_singleton(self):
        """测试 get_settings 返回单例"""
        s1 = get_settings()
        s2 = get_settings()
        assert s1 is s2
