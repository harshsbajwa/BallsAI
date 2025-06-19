import React, { useState } from "react";
import {
  View,
  Text,
  TextInput,
  FlatList,
  TouchableOpacity,
} from "react-native";
import { useQuery } from "@tanstack/react-query";
import { useRouter } from "expo-router";
import { apiClient } from "@/lib/api-client";
import { Loading } from "./Loading";
import { Player } from "@nba/api/types";

export function PlayerSearchCard() {
  const [searchQuery, setSearchQuery] = useState("");
  const router = useRouter();

  const { data: searchResults, isLoading } = useQuery({
    queryKey: ["players", "search", searchQuery],
    queryFn: () => apiClient.searchPlayers(searchQuery, 5),
    enabled: searchQuery.length >= 2,
  });

  const handlePlayerPress = (player: Player) => {
    setSearchQuery("");
    router.push(`/player/${player.person_id}`);
  };

  return (
    <View className="bg-white rounded-lg shadow-sm p-4">
      <Text className="text-lg font-bold text-gray-900 mb-4">
        Search Players
      </Text>

      <TextInput
        className="border border-gray-300 rounded-lg px-4 py-3 text-base"
        placeholder="e.g., LeBron James"
        value={searchQuery}
        onChangeText={setSearchQuery}
        autoCapitalize="words"
      />

      {searchQuery.length >= 2 && (
        <View className="mt-4">
          {isLoading ? (
            <Loading text="Searching..." />
          ) : searchResults?.items.length === 0 ? (
            <Text className="text-gray-500 text-center py-4">
              No players found
            </Text>
          ) : (
            <FlatList
              data={searchResults?.items}
              keyExtractor={(item) => item.person_id.toString()}
              renderItem={({ item }) => (
                <TouchableOpacity
                  onPress={() => handlePlayerPress(item)}
                  className="py-3 border-b border-gray-100 last:border-b-0"
                >
                  <Text className="font-semibold text-gray-900">
                    {item.full_name}
                  </Text>
                  <Text className="text-sm text-gray-500">
                    {item.primary_position}
                    {item.draft_year ? ` • Draft ${item.draft_year}` : ""}
                  </Text>
                </TouchableOpacity>
              )}
              showsVerticalScrollIndicator={false}
              scrollEnabled={false}
            />
          )}
        </View>
      )}
    </View>
  );
}